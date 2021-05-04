import multiprocessing
import argparse
import time
from numpy.lib.utils import lookfor
from torch.nn import parameter
import PPO
import socket
import threading
import multiprocessing as mp
from utils import MessageStream
import pickle
import logging
import torch
import scipy.signal
import numpy as np
logger = logging.getLogger('Training')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s: %(message)s'))
logger.addHandler(ch)
parser = argparse.ArgumentParser(
    description='Multi-Robot Collision Avoidance with Local Sensing via'
    'Deep Reinforcement Learning')

parser.add_argument(
    '--train', default=True, type=bool, help='train or test')
parser.add_argument(
    '--num_agents', default=20, type=int, help='number of robots')
parser.add_argument(
    '--num_obstacles', default=0, type=int, help='number of obstacles')
parser.add_argument(
    '--agent_radius', default=0.12, type=float, help='radius of the robot')
parser.add_argument(
    '--max_vx', default=1.0, type=float, help='max vx')
parser.add_argument(
    '--env_size', default=5., type=float, help='size of environment')


parser.add_argument(
    '--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument(
    '--lamda', default=0.95, type=float, help='gae')

# ppo
parser.add_argument(
    '--kl_target', default=0.0015, type=float)
parser.add_argument(
    '--beta', default=1., type=float)
parser.add_argument(
    '--eta', default=50., type=float)
parser.add_argument(
    '--actor_lr', default=2e-5, type=float)
parser.add_argument(
    '--clip_ratio', default=0.2, type=float)
parser.add_argument(
    '--actor_epochs', default=20, type=int)
parser.add_argument(
    '--critic_lr', default=1e-3, type=float)
parser.add_argument(
    '--critic_epochs', default=20, type=int)
parser.add_argument(
    '--lr_multiplier', default=1., type=float)

parser.add_argument(
    '--seed', default=333, type=int, help='random seed')
parser.add_argument(
    '--test_var', default=0., type=float, help='variance for test')

parser.add_argument(
    '--train_max_steps',
    default=3000000,
    type=int,
    help='max timesteps of the whole training')
parser.add_argument(
    '--batch_max_steps',
    default=8000,
    type=int,
    help='max timesteps of a batch for updating')
parser.add_argument(
    '--episode_max_steps',
    default=400,
    type=int,
    help='max timesteps of an episode')
parser.add_argument(
    '--train_max_iters',
    default=4000,
    type=int,
    help='maximum training iterations')
parser.add_argument(
    '--load_network',
    default=False,
    type=bool,
    help='whether to load pretrained networks')
parser.add_argument(
    '--train_batch_size',
    default=100,
    type=int,
    help='maximum training batch size')
parser.add_argument(
    '--train_type',
    default=0,
    type=int,
    help='maximum training batch size')
#0:ASP
#1:SP
#2:BSP
#3:m-nsp
def parameter_server_policy(policy_queue,model_queue,gathered_weight,isupdate,num_robots):
    weight=[]
    lr=[]
    while True:
        new_wight,new_lr=gathered_weight.get()
        lr.append(new_lr)
        if new_wight!=None:
            weight.append(new_wight)
        if len(lr)<num_robots-1:
            continue
        if len(weight)==0:
            for _ in range(num_robots-1):
                isupdate.put("ok")
            lr=[]
            continue
        all_weight=[]
        for i in range(len(weight[0])):
            tag=weight[0][i]
            for j in range(1,len(weight)):
                tag=tag+weight[j][i]
            all_weight.append(tag/len(weight))
        learning_rate=[]
        for i in lr:
            if i!=0:
                learning_rate.append(i)
        learning_rate=np.mean(learning_rate)
        # learning_rate=new_lr
        # all_weight=new_wight
        policy_net,policy_net_optimizer=policy_queue.get()
        policy_net_optimizer.param_groups[0]["lr"]=learning_rate
        for p,g in zip(policy_net.parameters(),all_weight):
            p.grad=g
        policy_net_optimizer.step()
        policy_net_optimizer.zero_grad()
        policy_queue.put((policy_net,policy_net_optimizer))
        model_queue.put(policy_net)
        for _ in range(num_robots-1):
            isupdate.put("ok")
        weight=[]
        lr=[]

def parameter_server_value(value_queue,gathered_weight,isupdate,num_robots):
    weight=[]
    while True:
        weight.append(gathered_weight.get())
        if len(weight)<num_robots-1:
            continue
        all_weight=[]
        for i in range(len(weight[0])):
            tag=weight[0][i]
            for j in range(1,len(weight)):
                tag=tag+weight[j][i]
            all_weight.append(tag/len(weight))
        #all_weight=gathered_weight.get()
        value_net,value_net_optimizer=value_queue.get()
        for p,g in zip(value_net.parameters(),all_weight):
            p.grad=g
        value_net_optimizer.step()
        value_net_optimizer.zero_grad()
        value_queue.put((value_net,value_net_optimizer))
        for _ in range(num_robots-1):
            isupdate.put("ok")
        weight=[]
def each_parameter_server(msg_stream,train_queue,args,policy_queue,value_queue,idx,istraining,gathered_wight_policy,isupdate_policy,gathered_wight_value,isupdate_value):
    while(True):
        msg=pickle.loads(msg_stream.recv())
        #logger.info(f'get order '+msg)
        if msg=='ready-data':
            istraining[idx]=False
            #logger.info(f"worker {idx} training end")
            msg_stream.send(pickle.dumps(train_queue.get()))
            istraining[idx]=True
            #logger.info(f"worker {idx} training start")
        if msg=='ready-args':
            msg_stream.send(pickle.dumps(args))
        if msg=='ask-policy':
            policy_net,policy_net_optimizer=policy_queue.get()
            policy_queue.put((policy_net,policy_net_optimizer))
            #todo:performance of dirty way
            msg_stream.send(pickle.dumps((policy_net,bytes(40*1024*1024))))
        if msg=='ask-value':
            value_net,value_net_optimizer=value_queue.get()
            value_queue.put((value_net,value_net_optimizer))
            #todo:performance of dirty way
            msg_stream.send(pickle.dumps((value_net,bytes(40*1024*1024))))
        if msg=='send-policy':
            weight,lr,_=pickle.loads(msg_stream.recv())
            gathered_wight_policy.put((weight,lr))
            msg=isupdate_policy.get()
            assert msg=="ok"
            msg_stream.send(pickle.dumps("ok"))
        if msg=='send-value':
            weight,_=pickle.loads(msg_stream.recv())
            gathered_wight_value.put(weight)
            msg=isupdate_value.get()
            assert msg=="ok"
            msg_stream.send(pickle.dumps("ok"))
        if msg=="early-break":
            gathered_wight_policy.put((None,0))
            msg=isupdate_policy.get()
            assert msg=="ok"
            msg_stream.send(pickle.dumps("ok"))
        #logger.info("finish order")
def discount(rewards, gamma):
    assert rewards.ndim >= 1
    # rewards[::-1]: reverse rewards, from n to 0
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]
def train_preprocess(experience_queue,train_queue,num_robots,num_obstacles,policy_queue,value_queue,record,args):
    iteration=0
    obs_inputs=[]
    acts=[]
    rewards=[]
    max_length=8000
    location=[]
    advs=np.array([])
    rets=np.array([])
    while True:
        datas,crahed,length=experience_queue.get()
        if datas==[]:
            continue
        for i in range(num_robots):
            for j in range(length[i]):
                data=datas[j]
                #perception_all=[]
                obstacle_all=[]
                #for j in range(num_robots):
                #    perception_all=perception_all+data[1][j]
                for k in range(num_obstacles):
                    obstacle_all=obstacle_all+data[2][k]
                obs_inputs.append(data[0][i]+data[1][i]+obstacle_all)
                acts.append(data[3][i])
                if j==0:
                    rewards.append(0.0)
                else:
                    rewards.append(data[4][i])
            location.append((length[i],crahed[i]))
        if len(rewards)>max_length:
            batch_size=len(rewards)//(num_robots-1)
            value_net,value_net_optimizer=value_queue.get()
            value_queue.put((value_net,value_net_optimizer))
            baseline_value = value_net(obs_inputs).detach().numpy()
            baseline_value = np.reshape(np.array(baseline_value), -1)
            rewards = np.array(rewards)
            start=0
            each=[]
            crashed_time=0
            for locat in location:
                end=start+locat[0]
                rets =np.append(rets, discount(rewards[start:end], args.gamma))
                b = np.append(baseline_value[start:end],0 if locat[1] else baseline_value[end-1])
                if locat[1]:
                    crashed_time=crashed_time+1
                deltas = rewards[start:end] + args.gamma*b[1:]-b[:-1]
                advs = np.append(advs,discount(deltas, args.gamma * args.lamda))
                each.append(rewards[start:end].sum())
                start=end
            advs=(advs-advs.mean())/(advs.std()+1e-6)
            for i in  range(num_robots-1):
                start=i*batch_size
                end=(i+1)*batch_size
                train_queue.put((obs_inputs[start:end],acts[start:end],advs[start:end],rets[start:end]))
            #train_queue.put((obs_inputs,acts,advs,rets))
            reward=np.array(each).mean()
            logger.info(f"all {len(location)} paths , {crashed_time} crash.")
            logger.info(f"average reward is {reward} , iteration {iteration}.")
            iteration=iteration+1
            obs_inputs=[]
            acts=[]
            rewards=[]
            location=[]
            advs=np.array([])
            rets=np.array([])
        if record:# and iteration%20==0:
            policy_net,policy_net_optimizer=policy_queue.get()
            policy_queue.put((policy_net,policy_net_optimizer))
            value_net,value_net_optimizer=value_queue.get()
            value_queue.put((value_net,value_net_optimizer))
            torch.save(policy_net.state_dict(), f'./model/pretrain/policy_{iteration}')
            torch.save(value_net.state_dict(), f'./model/pretrain/value_{iteration}')

def train(experience_queue: multiprocessing.Queue, model_queue: multiprocessing.Queue, num_robots, num_obstacles,ps_ip,ps_port,load,load_iteration,record,istraining, cmdargs: list = []):
    """
    Use `experience_queue` to retrieve experiences from the leader. One sample each time.
    Use `model_queue` to send newest model to the leader.

    Put a model to the `model_queue` immediately after start.
    Please write asynchronous code wisely, e.g. don't assume the leader can retrieve the model fast enough without blocking you.
    """
    args = parser.parse_args(cmdargs)
    policy_net=PPO.Policy_Net(num_robots,num_obstacles)
    if load:
        policy_net.load_state_dict(torch.load(f'./model/move/policy_{load_iteration}'))
        policy_net.reset()
    policy_net=policy_net.double()
    policy_net_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.actor_lr * args.lr_multiplier)
    policy_net_optimizer.zero_grad()
    model_queue.put(policy_net)
    Value_net=PPO.Value_Net(num_robots,num_obstacles)
    if load:
        Value_net.load_state_dict(torch.load(f'./model/move/value_{load_iteration}'))
    Value_net=Value_net.double()
    value_net_optimizer = torch.optim.Adam(Value_net.parameters(), lr=args.critic_lr)
    value_net_optimizer.zero_grad()
    # parameter-server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ps_ip, ps_port))
    sock.listen()
    policy_queue = mp.Queue(maxsize=1)
    policy_queue.put((policy_net,policy_net_optimizer))
    value_queue = mp.Queue(maxsize=1)
    value_queue.put((Value_net,value_net_optimizer))
    proc=[]
    train_queue=mp.Queue(maxsize=10000)
    gathered_weight_value=mp.Queue(maxsize=100)
    isupdate_value=mp.Queue(maxsize=100)
    gathered_weight_policy=mp.Queue(maxsize=100)
    isupdate_policy=mp.Queue(maxsize=100)
    t=threading.Thread(target=train_preprocess,args=(experience_queue,train_queue,num_robots,num_obstacles,policy_queue,value_queue,record,args))
    proc.append(t)
    t=threading.Thread(target=parameter_server_value,args=(value_queue,gathered_weight_value,isupdate_value,num_robots))
    proc.append(t)
    t=threading.Thread(target=parameter_server_policy,args=(policy_queue,model_queue,gathered_weight_policy,isupdate_policy,num_robots))
    proc.append(t)
    for i in range(num_robots-1):
        client_sock, (client_ip, _) = sock.accept()
        msg_stream=MessageStream(client_sock,mutex=True)
        t=threading.Thread(target=each_parameter_server,args=(msg_stream,train_queue,args,policy_queue,value_queue,i,istraining,gathered_weight_policy,isupdate_policy,gathered_weight_value,isupdate_value))
        proc.append(t)
    for t in proc:
        t.start()
    for t in proc:
        t.join()

    
