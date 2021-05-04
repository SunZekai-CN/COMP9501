"""
This oracle interfaces with the simulator. To be specific, it:
1. gets sensor data from the simulator and sends them to robots.
2. gets actuator data from the robots and sends them to the simulator.
3. controls the obstacles in the simulator.
"""

from collections import Counter
import sys
import time
import socket
from utils import MessageStream, calibrate_freq
import copy
import yaml
import time
import pickle
import rclpy
import threading
import math
from functools import partial
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry


class ObstacleController(threading.Thread):
    def __init__(self, num_obstacles):
        super().__init__()
        self.ros2_node = rclpy.create_node('obctl')
        self.publishers = [self.ros2_node.create_publisher(Twist, f'/obstacle_{i}/cmd_vel', 10) for i in range(num_obstacles)]

    def run(self):
        while False:
            time.sleep(0.1)  # this just need to be smaller that the watchdog timeout
            msg = Twist(
                linear=Vector3(x=1.0, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=1.0)
            )
            for p in self.publishers:
                p.publish(msg)


class SensorBridge(threading.Thread):
    def __init__(self, mstreams):
        super().__init__()
        self.mstreams = mstreams
        self.ros2_node = rclpy.create_node('sensor_bridge')

        self.datas = [None]*len(mstreams)
        self.datas_lock = threading.Lock()

        def handle_sensor_data(i, msg):
            # Convert the format
            # [x, y, theta, vx, vy]
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            ori = msg.pose.pose.orientation
            theta = math.atan2(2.0*(ori.w*ori.z + ori.x*ori.y), 1.0 - 2.0*(ori.y*ori.y + ori.z*ori.z))
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            data = [x, y, theta, vx, vy]

            with self.datas_lock:
                self.datas[i] = data
        self.subscribers = [self.ros2_node.create_subscription(Odometry, f'/robot_{i}/base_pose_ground_truth', partial(handle_sensor_data, i), 10) for i in range(len(mstreams))]

    def run(self):
        spin_thread = threading.Thread(target=lambda: rclpy.spin(self.ros2_node))
        spin_thread.start()

        while True:
            calibrate_freq(30, 0)
            with self.datas_lock:
                datas = copy.copy(self.datas)
            for i, data in enumerate(datas):
                if data is None:
                    continue
                self.mstreams[i].send(pickle.dumps(data))

        spin_thread.join()

class ActuatorBridge(threading.Thread):
    def __init__(self, mstreams):
        super().__init__()
        self.mstreams = mstreams
        self.ros2_node = rclpy.create_node('actuator_bridge')

        self.publishers = [self.ros2_node.create_publisher(Twist, f'/robot_{i}/cmd_vel', 10) for i in range(len(mstreams))]

    def run(self):
        def f(mstream, publisher):
            while True:
                try:
                    tmp = pickle.loads(mstream.recv())
                    v_t, v_r = tmp
                    assert not math.isnan(v_t) and not math.isnan(v_r)

                    # [translational speed, rotational speed (rad)]
                    msg = Twist(
                        linear=Vector3(x=v_t, y=0.0, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=v_r),
                    )

                    publisher.publish(msg)
                except:
                    print(f'action type wrong or nan, {tmp}', flush=True)
        threads = [threading.Thread(target=f, args=(ms,p)) for ms, p in zip(self.mstreams, self.publishers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    config_path = sys.argv[1]

    # parse config
    with open(config_path) as conf:
        config = yaml.load(conf, yaml.FullLoader)
    robot_ips = config['sim_ips']
    my_port = config['oracle_port']

    # connect with other nodes
    my_ip = robot_ips[0]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((my_ip, my_port))
    sock.listen()

    msg_streams = []
    connected_ips = []
    for _ in range(len(robot_ips)):
        client_sock, (client_ip, _) = sock.accept()
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        msg_streams.append(MessageStream(client_sock))
        connected_ips.append(client_ip)
        print(f'{client_ip} connected', flush=True)
    assert Counter(connected_ips) == Counter(robot_ips)
    msg_streams = [s for s, _ in sorted(zip(msg_streams, connected_ips), key=lambda t: robot_ips.index(t[1]))]

    rclpy.init(args=['--ros-args', '--log-level', 'info'])

    obstacle_controller = ObstacleController(5)
    obstacle_controller.start()

    sensor_bridge = SensorBridge(msg_streams)
    sensor_bridge.start()

    actuator_bridge = ActuatorBridge(msg_streams)
    actuator_bridge.start()

    for t in [obstacle_controller, sensor_bridge, actuator_bridge]:
        t.join()
