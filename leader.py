# Copyright 2017 The DRLCA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


import utils
import time
import yaml
import time
import array
import copy
import pickle
import rclpy
from std_msgs.msg import UInt8MultiArray
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import threading
import logging
import math
import sys
from functools import partial
import queue
import numpy as np
import multiprocessing as mp
from train import train
import torch
import itertools
logger = logging.getLogger('WorkerCycle')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s: %(message)s'))
logger.addHandler(ch)


F = 30


class Leader(threading.Thread):
    def __init__(self, num_robots, num_obstacles, init_positions, targets, max_steps=F*15, max_dist=20, reset_dist=15):
        super().__init__()

        self.num_robots = num_robots
        self.num_obstacles = num_obstacles
        self.init_positions = init_positions
        self.targets = targets

        self.max_steps = max_steps
        self.max_dist = max_dist
        self.reset_dist = reset_dist

        self.perception_queues = [queue.Queue(maxsize=1) for _ in range(num_robots)]

        self.ros2_node = rclpy.create_node('leader')
        self.reset_service = self.ros2_node.create_client(Empty, 'reset_positions')

        # get perceptions
        def on_perception_subscription(i, msg):
            ts_receive = time.time()
            perc, ts, _ = pickle.loads(msg.data)

            # always keep newest one
            # this piece of code works because I'm the only producer
            if self.perception_queues[i].full():
                self.perception_queues[i].get_nowait()
            self.perception_queues[i].put((perc, ts))

            #logger.info(f'perception delay of {i} {(ts_receive - ts)*1e3:.3f}ms')
        self.perception_subscribers = [self.ros2_node.create_subscription(UInt8MultiArray, f'{"urg_" if i != 0 else ""}perception_{i}', partial(on_perception_subscription, i), utils.get_ros_qos_profile()
            ) for i in range(num_robots)]

        # get obstacle states
        # Note that in the paper of Han et. al., it was stated that 'the obstacles state is aggregated from the observation of all robots'.
        # It's not clear how they are aggregated.
        # Here we simply get it from the simulator.
        # The communication still exists for aggregating robot states.
        self.obstacle_states = [None]*num_obstacles
        self.obstacle_states_lock = threading.Lock()
        def handle_obstacle_state(i, msg):
            # Convert the format
            # [x, y, vx, vy, radius]
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            radius = 0.24

            with self.obstacle_states_lock:
                self.obstacle_states[i] = [x, y, vx, vy, radius]
        self.obstacle_subscribers = [self.ros2_node.create_subscription(Odometry, f'/obstacle_{i}/base_pose_ground_truth', partial(handle_obstacle_state, i), 10 
            ) for i in range(num_obstacles)]

        self.spin_thread = threading.Thread(target=lambda: rclpy.spin(self.ros2_node))

        # update models from the trainer
        self.model_queue = mp.Queue(maxsize=1)
        self.model = None
        self.model_lock = threading.Lock()
        def update_model_task():
            while True:
                model = self.model_queue.get()
                with self.model_lock:
                    self.model = model
        self.update_model_thread = threading.Thread(target=update_model_task)
 
        # make decisions, and also maintain replay memory
        self.action_publishers = [self.ros2_node.create_publisher(
            UInt8MultiArray,
            # f'{"urg_" if i != 0 else ""}action_{i}',
            f'action_{i}',
            utils.get_ros_qos_profile()
        ) for i in range(num_robots)]
        # the replay memory
        # each record is of the following form:
        # record := robot_targets + robot_states + obstacle_states + robot_actions + rewards
        # these xxxs are lists
        # robot_target := [x, y]
        # robot_state := [x, y, orientation, vx, vy]
        # obstacle_state := [x, y, vx, vy, radius]
        # robot_action := [vt, vr]
        self.experience_queue = mp.Queue(maxsize=10000)
        manager=mp.Manager()
        self.istraining=manager.list([False for _ in range(self.num_robots-1)])
        def get_perceptions():
            # always use newest perception of all the robots
            rets = [q.get() for q in self.perception_queues]
            for i, q in enumerate(self.perception_queues):
                try:
                    rets[i] = q.get_nowait()
                except:
                    pass
            return zip(*rets)
        def control_task():
            # the first several samples are unstable, skip them
            time.sleep(5)
            seed_iter=itertools.count()
            def run_one_episode():
                # There is no magic in the world to reset the env, so we do it by ourselves
                while True:
                    perceptions, perception_times = get_perceptions()

                    actions = [[0.0, 0.0]]*self.num_robots
                    need_more_actions = False
                    for i in range(num_robots):
                        d_vec = np.subtract(self.init_positions[i], perceptions[i][:2])
                        my_theta = perceptions[i][2]
                        def get_angle(v1, v2):
                            dot = v1[0]*v2[0] + v1[1]*v2[1]
                            det = v1[0]*v2[1] - v1[1]*v2[0]
                            return math.atan2(det, dot)
                        target_theta = get_angle([1, 0], d_vec)
                        if np.linalg.norm(d_vec) > 0.5 and 2*math.pi-0.1> abs(my_theta - target_theta) > 0.1:
                            need_more_actions = True
                            d_theta = target_theta - my_theta
                            if abs(d_theta) > math.pi:
                                d_theta = (2*math.pi - abs(d_theta)) * (-1 if d_theta > 0 else 1)
                            actions[i] = (0.0, 60.0 * np.sign(d_theta))
                        elif np.linalg.norm(d_vec) > 0.5:
                            need_more_actions = True
                            # move towards the target
                            actions[i] = (60.0, 0.0)  # Turn around
                        elif abs(perceptions[i][2]) > 0.1:
                            need_more_actions = True
                            d_theta = -my_theta
                            if abs(d_theta) > math.pi:
                                d_theta = (2*math.pi - abs(d_theta)) * (-1 if d_theta > 0 else 1)
                            actions[i] = (0.0, 60.0 * np.sign(d_theta))

                    ts = time.time()
                    for p, a, pts in zip(self.action_publishers, actions, perception_times):
                        msg = UInt8MultiArray()
                        msg.data = array.array('B', pickle.dumps((a, ts, pts)))
                        p.publish(msg)
                    if not need_more_actions:
                        break

                prev_exp = None
                experiences = []
                crashed=[False for i in range(self.num_robots)]
                torch.manual_seed(seed_iter.__next__())
                length=[-1 for _ in range(self.num_robots)]
                start=time.monotonic()
                not_training=False
                if not any(self.istraining):
                    not_training=True
                # if not_training:
                #     logger.info(f"start moving")
                # else:
                #     logger.info(f"start moving during training")
                while True:
                    # this is synchronous
                    # both synchronous and asynchronous will be negatively influenced by the latency jitter, but in different ways
                    # asynchronous is more suitable for such systems, but let's use the synchronous one to show the problem for now
                    end=time.monotonic()
                    perceptions, perception_times = get_perceptions()
                    #loop_start=time.monotonic()
                    with self.obstacle_states_lock:
                        obstacle_states = copy.copy(self.obstacle_states)

                    targets = self.targets

                    with self.model_lock:
                        model = self.model

                    if any([p is None for p in perceptions]) or any([p is None for p in obstacle_states]) or model is None:
                        continue

                    actions = self.generate_actions(model, targets, perceptions, obstacle_states)
                    if np.any(np.isnan(actions)):
                        logger.error('nan from the model, exit')
                        exit(1)
                    # Check if they are running too far away
                    episode_end = False
                    for i in range(num_robots):
                        if np.linalg.norm(np.subtract(targets[i], perceptions[i][:2])) > self.max_dist:
                            episode_end = True
                            break
                    episode_end = episode_end or (end-start)>self.max_steps/F
                    if episode_end:
                        break
                    for i in range(self.num_robots):
                        if crashed[i]:
                            actions[i]=[0.0,0.0]
                    ts = time.time()
                    for p, a, pts in zip(self.action_publishers, actions, perception_times):
                        msg = UInt8MultiArray()
                        msg.data = array.array('B', pickle.dumps((a, ts, pts)))
                        p.publish(msg)

                    # this part of code must be fast enough to avoid blocking future control tasks
                    if prev_exp is not None:
                        rewards = [0]*self.num_robots
                        t_begin = time.monotonic()
                        for i in range(self.num_robots):
                            # rg
                            d_target = np.linalg.norm(np.subtract(targets[i], perceptions[i][:2]))
                            d_target_prev = np.linalg.norm(np.subtract(prev_exp[0][i], prev_exp[1][i][:2]))
                            if d_target < 0.1:
                                rg = 5.0
                            elif 0.1 < d_target < 0.4:
                                rg = 1.0 + d_target * 0.5
                            else:
                                rg = 10 * (d_target_prev - d_target)
                            # rc
                            # min robot distance
                            radius = 0.24
                            min_dist = 1.0e9
                            for j in range(self.num_obstacles):
                                if min_dist <= 2.05 * radius:
                                    break
                                dist = np.linalg.norm(np.subtract(perceptions[i][:2], obstacle_states[j][:2]))
                                min_dist = min(min_dist, dist)
                            if min_dist <= 2.05 * radius:
                                rc = -10.0
                                if crashed[i]==False:
                                    length[i]=len(experiences)+1
                                    crashed[i]=True  
                            else:
                                rc=0.0
                            rewards[i] = rg  + rc
                            #logger.info(f' reward {rewards[i]},{d_target},{rg},{rc},{prev_exp}')
                            # rewards[i] = -(abs(prev_exp[3][i][0])+abs(prev_exp[3][i][1]))
                        t_end = time.monotonic()
                        #logger.debug(f'spent {(t_end - t_begin)*1e3:.3f}ms in computing reward')
                        #logger.debug(f"rewards: {rewards}")
                        # store it
                        # always drop the oldest item
                        # again, this works because I'm the only producer
                        experiences.append((*prev_exp, rewards))

                    prev_exp = (targets, perceptions, obstacle_states, actions)
                    if all(crashed):
                        break
                    #loop_end=time.monotonic()
                    #logger.info(f"cycle time {(loop_end-loop_start)*1e+3}")
                # record the experiences in this epoch
                # if not_training:
                #     logger.info(f"end moving")
                # else:
                #     logger.info(f"end moving during training")
                # #logger.info(f'finished one epoch')
                if self.experience_queue.full():
                    self.experience_queue.get_nowait()
                for i in range(len(length)):
                    if length[i]==-1:
                        length[i]=len(experiences)
                if not_training:
                    self.experience_queue.put((experiences,crashed,length))
                all_reward=[]
                for i in range(num_robots):
                    each_reward=[]
                    for j in range(length[i]):
                        if j==0:
                            each_reward.append(0.0)
                        else:
                            each_reward.append(experiences[j][4][i])
                    all_reward.append(np.sum(each_reward))
                logger.info(f"average reward is {np.mean(all_reward)}")
                    

            while True:
                run_one_episode()

        self.control_thread = threading.Thread(target=control_task)

        self.training_process = mp.Process(target=train, args=(self.experience_queue, self.model_queue, self.num_robots, self.num_obstacles,'0.0.0.0',4097,True,150,False,self.istraining))

    def generate_actions(self, model, targets, perceptions, obstacle_states):
        #perceptions_all=[]
        obstacle_states_all=[]
        action=[]
        #for i in range(self.num_robots):
        #    perceptions_all=perceptions_all+perceptions[i]
        for i in range(self.num_obstacles):
            obstacle_states_all=obstacle_states_all+obstacle_states[i]
        data=[]
        for i in range(self.num_robots):
            data.append(targets[i]+perceptions[i]+obstacle_states_all)
        log_vars, action_mean = model(data)
        for i in range(self.num_robots):
            sampled_act = action_mean[i]+torch.exp(log_vars)*torch.randn((2,))
            sampled_act = sampled_act.detach().numpy()
            sampled_act[0]=np.clip(sampled_act[0],0.0,1.0)
            sampled_act[1]=np.clip(sampled_act[1],-1.0,1.0)
            action.append(sampled_act)
        return np.array(action).tolist()

    def run(self):
        threads = [self.spin_thread, self.control_thread, self.update_model_thread, self.training_process]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

if __name__ == "__main__":
    with open(sys.argv[1]) as conf:
        config = yaml.load(conf, yaml.FullLoader)
    sim_ips = config['sim_ips']
    num_obstacles = config['num_obstacles']
    targets = [
         [0.0, 5.0], [-4.8, 1.5], [-4.7, -5], [4.7, -5], [4.8, 1.5],  # roughly a star
         [1.0, 5.0], [-5.8, 1.5], [-5.7, -5], [5.7, -5], [5.8, 1.5]][:len(sim_ips)]
    #targets = [[0.0, 0.0]]*len(sim_ips)
    init_positions = [[-6.0, 0.0], [0.0, -6.0], [6.0, 0.0], [6.0, 6.0], [-6.0, 6.0],[-6.0,-6.0],[6.0,-6.0]][:len(sim_ips)]

    rclpy.init(args=['--ros-args', '--log-level', 'info'])

    leader = Leader(len(sim_ips), num_obstacles, init_positions, targets)
    leader.start()
    leader.join()
