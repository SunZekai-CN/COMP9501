import sys
import socket

from numpy.core.fromnumeric import mean
from sklearn.utils.validation import _check_sample_weight
from torch.nn.modules.module import T
from utils import MessageStream
import torch
import pickle
import numpy as np
import scipy.signal
from sklearn.utils import shuffle
from PPO import compute_policy_net_loss


def discount(rewards, gamma):
    assert rewards.ndim >= 1
    # rewards[::-1]: reverse rewards, from n to 0
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]


def update_policy_net(obs_inputs, acts, advs, leader_msg_stream, args):
    leader_msg_stream.send(pickle.dumps("ask-policy"))
    policy_net,_ = pickle.loads(leader_msg_stream.recv())
    old_log_vars_np, old_means_np = policy_net(obs_inputs)
    old_log_vars_np = old_log_vars_np.detach()
    old_means_np = old_means_np.detach()
    compute_loss = compute_policy_net_loss( args.actor_lr * args.lr_multiplier, args.kl_target, 2, False, args.clip_ratio)
    print("lr", args.actor_lr * args.lr_multiplier, flush=True)
    early_break=False
    for e in range(args.actor_epochs):
        if early_break:
            leader_msg_stream.send(pickle.dumps("early-break"))
            msg = pickle.loads(leader_msg_stream.recv())
            assert msg == 'ok'
            continue
        leader_msg_stream.send(pickle.dumps("ask-policy"))
        policy_net,_ = pickle.loads(leader_msg_stream.recv())
        log_vars, action_mean = policy_net(obs_inputs)
        loss, kl = compute_loss(torch.tensor(acts), action_mean, log_vars, old_means_np, old_log_vars_np, torch.tensor(advs), args.beta, args.eta)
        print("actor losss",loss,flush=True)
        loss.backward()
        weight=[]
        for p in policy_net.parameters():
            weight.append(p.grad)
        leader_msg_stream.send(pickle.dumps("send-policy"))
        leader_msg_stream.send(pickle.dumps((weight,args.actor_lr * args.lr_multiplier,bytes(40*1024*1024))))
        msg = pickle.loads(leader_msg_stream.recv())
        assert msg == 'ok'
        if kl > args.kl_target * 4:  # early stopping
            early_break=True

    if kl > args.kl_target * 2:
        args.beta = np.minimum(35, 1.5 * args.beta)
        if args.beta > 30 and args.lr_multiplier > 0.1:
            args.lr_multiplier /= 1.5
    elif kl < args.kl_target / 2.0:
        args.beta = np.maximum(1.0 / 35.0, args.beta / 1.5)
        if args.beta < (1.0 / 30.0) and args.lr_multiplier < 10:
            args.lr_multiplier *= 1.5
    print("log_vars", log_vars, flush=True)
    print("update policy net", flush=True)


def update_value_net(obs_scan, rets, leader_msg_stream, args, last,value_net_trained):
    if last ==[]:
        obs_scan_train, rets_train = obs_scan, rets
        last.append((obs_scan, rets))
    else:
        obs_scan_train = np.concatenate([obs_scan, last[0][0]]).tolist()
        rets_train = np.concatenate([rets, last[0][1]]).tolist()
        last[0]=(obs_scan, rets)
    if value_net_trained:
        epoches=args.critic_epochs
    else:
        epoches=args.critic_epochs*10
    for e in range(epoches):
        leader_msg_stream.send(pickle.dumps("ask-value"))
        value_net,_ = pickle.loads(leader_msg_stream.recv())
        #batch = 256
        (obs_scan_train, rets_train) = shuffle(obs_scan_train, rets_train)
        # batch_num=len(obs_scan_train)//batch
        # for i in range(batch_num):
        #     start=i*batch
        #     end=(i+1)*batch
        value = value_net(obs_scan_train)
        loss = torch.mean(
                0.5*torch.square(value - torch.tensor(rets_train).view(-1, 1)))
            # print("value",value)
            # print("rets",torch.tensor(rets_train[start:start+batch]).view(-1, 1))
            # print("value loss all",torch.square(value - torch.tensor(rets_train[start:start+batch]).view(-1, 1)))
        print("value loss",loss,flush=True)
        loss.backward()
        weight=[]
        for p in value_net.parameters():
            weight.append(p.grad.detach())
        leader_msg_stream.send(pickle.dumps("send-value"))
        leader_msg_stream.send(pickle.dumps((weight,bytes(40*1024*1024))))
        msg = pickle.loads(leader_msg_stream.recv())
        assert msg == 'ok'
    print("update value net", flush=True)


def training_data_process(obs_inputs,acts,advs,rets,args, leader_msg_stream, value_net_trained, last):
    print("all prepared", flush=True)
    #obs_inputs=obs_inputs[:-1]
    # advantage=dr-baseline_value
    # advantage=(advantage-advantage.mean())/(advantage.std()+1e-6)
    #print("value net trained",value_net_trained)
    if value_net_trained:  # train acotr after trained critic
        update_policy_net(obs_inputs, acts, advs.tolist(),
                          leader_msg_stream, args)
    update_value_net(obs_inputs, rets.tolist(), leader_msg_stream, args, last,value_net_trained)


def train_worker(ps_ip, ps_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ps_ip, ps_port))
    leader_msg_stream = MessageStream(sock, mutex=True)
    print("conneted to parameter server", flush=True)
    value_net_trained = False
    leader_msg_stream.send(pickle.dumps("ready-args"))
    args = pickle.loads(leader_msg_stream.recv())
    print("get args", flush=True)
    i = 0
    last = []
    while(True):
        print(f'**********iteration {i} ************', flush=True)
        i = i+1
        leader_msg_stream.send(pickle.dumps("ready-data"))
        obs_inputs,acts,advs,rets = pickle.loads(leader_msg_stream.recv())
        print("get training data", len(obs_inputs), flush=True)
        training_data_process(obs_inputs,acts,advs,rets,args,
                              leader_msg_stream, value_net_trained, last)
        value_net_trained = True


if __name__ == '__main__':
    ps_ip = sys.argv[1]
    ps_port = int(sys.argv[2])
    train_worker(ps_ip, ps_port)
