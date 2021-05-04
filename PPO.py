from math import nan
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.testing._private.utils import requires_memory
from scipy.signal.waveforms import square
import torch
from torch import tensor
import torch.nn as nn
from torch.autograd import Variable
DUMMY = int(100*1000*1000/67)
def preprocess(x,num_robot,num_obstacles):
    output=4+4*(num_obstacles)
    output=np.array([[0.0 for i in range(output)] for j in range(len(x))])
    inputs=np.array(x)
    theta=inputs[:,4]
    cos_theta=np.cos(theta)
    sin_theta=np.sin(theta)
    output[:,0]=cos_theta*(inputs[:,0]-inputs[:,2])+sin_theta*(inputs[:,1]-inputs[:,3])
    output[:,1]=-sin_theta*(inputs[:,0]-inputs[:,2])+cos_theta*(inputs[:,1]-inputs[:,3])
    # for i in range(num_robot):
    #     output[:,2+i*5]=cos_theta*(inputs[:,7+i*5]-inputs[:,2])+sin_theta*(inputs[:,8+i*5]-inputs[:,3])
    #     output[:,3+i*5]=-sin_theta*(inputs[:,7+i*5]-inputs[:,2])+cos_theta*(inputs[:,8+i*5]-inputs[:,3])
    #     output[:,4+i*5]=inputs[:,9+i*5]-theta
    output[:,2]=cos_theta*inputs[:,5]+sin_theta*inputs[:,6]
    output[:,3]=-sin_theta*inputs[:,5]+cos_theta*inputs[:,6]
    start=2
    for i in range(num_obstacles):
        output[:,start+2+i*4]=cos_theta*(inputs[:,start+5+i*5]-inputs[:,2])+sin_theta*(inputs[:,start+6+i*5]-inputs[:,3])
        output[:,start+3+i*4]=-sin_theta*(inputs[:,start+5+i*5]-inputs[:,2])+cos_theta*(inputs[:,start+6+i*5]-inputs[:,3])
        output[:,start+4+i*4]=cos_theta*inputs[:,start+7+i*5]+sin_theta*inputs[:,start+8+i*5]
        output[:,start+5+i*4]=-sin_theta*inputs[:,start+7+i*5]+cos_theta*inputs[:,start+8+i*5]
    return output
        
        
class Policy_Net(nn.Module):
    def __init__(self,num_robots, num_obstacles):
        super(Policy_Net, self).__init__()
        self.fc1=nn.Linear(4+4*(num_obstacles),256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        #self.fc4=nn.Linear(64,DUMMY)
        #self.fc5=nn.Linear(DUMMY,1)
        #self.fc6=nn.Linear(DUMMY,1)
        self.fc4=nn.Linear(64,1)
        self.fc5=nn.Linear(64,1)
        self.logvar_speed =20
        self.num_robots=num_robots
        self.num_obstacles=num_obstacles
        self.log_vars_tensor=nn.Parameter(torch.zeros(self.logvar_speed,2,requires_grad=True).type(torch.FloatTensor))
    def forward(self, input):
        x = torch.tensor(preprocess(input,self.num_robots,self.num_obstacles))

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x)) 

        linear=torch.sigmoid(self.fc4(x))
        
        angular=torch.tanh(self.fc5(x))
    
        action_mean=torch.cat((linear,angular), dim=1) 

        log_vars=torch.sum(self.log_vars_tensor,axis=0)

        return log_vars,action_mean
    def reset(self):
        self.log_vars_tensor=nn.Parameter(torch.zeros(self.logvar_speed,2,requires_grad=True).type(torch.FloatTensor))


class Value_Net(nn.Module):
    def __init__(self,num_robots, num_obstacles):
        super(Value_Net, self).__init__()
        self.fc1=nn.Linear(4+4*(num_obstacles),128)
        self.fc2=nn.Linear(128,64)
        #self.fc3=nn.Linear(64,DUMMY)
        self.fc3=nn.Linear(64,1)
        self.num_robots=num_robots
        self.num_obstacles=num_obstacles

    def forward(self, scan):
        x=torch.tensor(preprocess(scan,self.num_robots,self.num_obstacles))

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) 
        value=self.fc3(x)
        return value
class compute_policy_net_loss(nn.Module):
    def __init__(self, lr_ph, kl_targ,act_dim,only_entropy,clip_ratio):
        super(compute_policy_net_loss, self).__init__()
        self.lr_ph=torch.tensor(lr_ph)
        self.kl_targ=torch.tensor(kl_targ)
        self.act_dim=torch.tensor(act_dim)
        self.only_entropy=only_entropy
        self.clip_ratio=clip_ratio
        self.entropy_coef=0.01

    def forward(self, act_ph, means, log_vars, old_means_ph, old_log_vars_ph, advantages_ph,beta_ph, eta_ph):
        self.beta_ph=torch.tensor(beta_ph)
        self.eta_ph=torch.tensor(eta_ph)
        # compute logprob
        entropy = 0.5 * (
                self.act_dim *
                (np.log(2 * np.pi) + 1)) + torch.sum(log_vars)
        if self.only_entropy:
            return entropy
        logp = torch.sum(
            -log_vars + -0.5 *
                torch.square(act_ph - means) / torch.square(torch.exp(log_vars)),
                axis=1)
        logp_old = torch.sum(
            -old_log_vars_ph + -0.5 *
                torch.square(act_ph - old_means_ph) / torch.square(torch.exp(old_log_vars_ph)),
                axis=1)
        # ratio=torch.exp(logp - logp_old)
        # min_adv=torch.clip(ratio,1-self.clip_ratio,1+self.clip_ratio)*advantages_ph
        # loss1=-torch.mean(torch.minimum(ratio*advantages_ph,min_adv))
        # loss2=-self.entropy_coef*entropy
        # actor_loss=loss1+loss2
        tag=-2*old_log_vars_ph+2*log_vars+torch.square(torch.exp(old_log_vars_ph - log_vars))+torch.square(means - old_means_ph) / torch.square(torch.exp(log_vars))
        kl = 0.5 *torch.mean(torch.sum(tag,axis=1)-self.act_dim)
        # compute entropy
        
        loss1 = -torch.mean(
            advantages_ph * torch.exp(logp - logp_old))
        loss2 = torch.mean(self.beta_ph * kl)
        loss3 = self.eta_ph * torch.square(
            torch.maximum(torch.tensor(0.0), kl - 2.0 * self.kl_targ))
        actor_loss = loss1 + loss2 + loss3
        #'advantages_ph', 'logp', 'logp_old', 'log_vars', 'act_ph', 'means', 'old_means_ph', 'old_log_vars_ph',
        for key in ['actor_loss', 'loss1','loss2','loss3','kl']:
            try:
                print(f'{key}={eval(key)}')
                #continue
            except:
                pass
        # kl= None
        return actor_loss,kl
