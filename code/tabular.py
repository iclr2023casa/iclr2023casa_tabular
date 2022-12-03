import torch.nn as nn
import torch
import torch.nn.functional as F
from grid_world import GridWorld
import numpy as np
import random
from functools import partial

import ray

device ='cpu'

def get_one_hot(x, h, w):
    y = torch.zeros([h*w])
    y[x]=1.0
    return y.to(device)

def get_model(h,w):
    return nn.Linear(h*w, 4, bias=False).to(device)

def get_model_v(h, w):
    return nn.Linear(h*w, 1, bias=False).to(device)

import gym

num_rows = 4
num_cols = 4
LR = 0.05
EPOCHS = 1000
MCSAMPLES = 20
CHI_BATCH = 16

def randomize(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



# behavior_policy = get_model(num_rows, num_cols)



def casa_q_pi(a, v, tau=1.0):
    p = torch.softmax(tau*a,dim=-1)
    p_sg = p.detach()
    q = a - (p_sg*a).sum(dim=-1,keepdim=True)+ v.detach()
    return q, p

def casa_mix_q_pi(a, v, logits, alpha, tau=1.0):
    log_p = torch.log_softmax(alpha * logits + (1 - alpha) * a * tau, dim=-1)
    p = torch.softmax(log_p,dim=-1)
    p_sg = p.detach()
    q = a - (p_sg*a).sum(dim=-1,keepdim=True)+ v.detach()
    return q, p

def log_mpo_pi(q, logits, alpha=0.5, tau=1.0):
    pi = torch.log_softmax(alpha*logits+(1-alpha)*q.detach()*tau,dim=-1)
    # pi_new = pi * torch.exp(q*tau) / (torch.exp(q*tau).sum())
    return pi

def muse_pi(adv, logits, c):
    pi = torch.softmax(logits)
    pi_new = pi * torch.exp(torch.clamp(adv,-c,c)) / (torch.exp(torch.clamp(adv,-c,c)).sum())
    return pi_new

def sample(a):
    p=torch.distributions.categorical.Categorical(logits=a)
    return p.sample()
# def sample_p(a)


def mc(env, p):
    done = False
    rs = []
    Ss = []
    As = []
    s = env.reset()
    i = 0
    while not done:
        o_s = get_one_hot(s, num_rows, num_cols)
        Ss.append(o_s)
        a = sample(p(o_s))
        s,r,done,_ = env.step(a)
        As.append(a)
        rs.append(r)
        i += 1
        if i >= num_rows*num_cols*10:
            break
    ret = 0.0
    datas = []
    for r,s,a in zip(reversed(rs),reversed(Ss),reversed(As)):
        ret+=r
        datas.append((s,a,ret))
    return datas, sum(rs)

def get_data(env, p, itr=1000):
    data = []
    rets = []
    for i in range(itr):
        new_data, ret = mc(env, p)
        data+=new_data
        rets.append(ret)
    return data, sum(rets)/len(rets)

def get_batch(data_p, bz=256, random=True):
    if random:
        f = np.random.permutation(len(data_p))
    else:
        f = list(range(len(data_p)))
    Ss = []
    As = []
    Rs = []
    for i in f[:bz]:
        s,a,r = data_p[i]
        Ss.append(s)
        As.append(a)
        Rs.append(r)
        # d.append(data_p[i])
    return torch.stack(Ss).to(device),torch.stack(As).to(device),torch.tensor(Rs).to(device)

def get_logp(Ss, p_policy):
    logits = p_policy(Ss)
    return logits.log_softmax(dim=-1)

def get_logp_grad_loss(Ss,As, p_policy):
    logits = p_policy(Ss)
    logp = logits.log_softmax(dim=-1)
    return logp.gather(-1,As.unsqueeze(1)).squeeze()

def get_q_grad_loss(Ss, As, q_func):
    qa = q_func(Ss).gather(-1,As.unsqueeze(1)).squeeze()
    return qa

def get_Q_loss(Ss,As,Rs,q_func):
    qa = q_func(Ss).gather(-1,As.unsqueeze(1)).squeeze()
    loss = F.mse_loss(qa,Rs, reduction='none')
    return loss

def get_J_loss(Ss,As,Rs,logp_func):
    log_p = logp_func(Ss).gather(-1,As.unsqueeze(1)).squeeze()
    return -(Rs*log_p)

def get_V_loss(Ss,Rs,v_func):
    return F.mse_loss(v_func(Ss).squeeze(), Rs, reduction='none')


@ray.remote
def baseline_exp(seed, out_file=None):
    randomize(seed)

    env = GridWorld(num_rows=num_rows,
                num_cols=num_cols)
    R,Chi,Beta=[],[],[]
    Q_net = get_model(num_rows, num_cols)
    P_net = get_model(num_rows, num_cols)
    Casa_net = get_model(num_rows, num_cols)
    Casa_net.load_state_dict(P_net.state_dict()) # align casa and policy
    Casa_v = get_model_v(num_rows, num_cols)
    optim = torch.optim.SGD(list(Q_net.parameters())+list(P_net.parameters()),lr= LR, weight_decay=0.0, momentum=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=1e-6, last_epoch=- 1, verbose=False)
    for epoch in range(EPOCHS):
        data, rs = get_data(env, P_net, MCSAMPLES)
        # print(epoch,'r=',rs,  )
        R.append(rs)
        Ss,As,Rs = get_batch(data,len(data),False)
        q_loss = get_Q_loss(Ss,As,Rs,Q_net)
        p_loss = get_J_loss(Ss,As,Rs,partial(get_logp, p_policy=P_net))
        # print(q_loss.shape,p_loss.shape)
        # print(q_loss.mean().item(),p_loss.mean().item())
        grad_q_no_flat = torch.autograd.grad(q_loss.mean(),Q_net.parameters())
        grad_q = grad_q_no_flat[0].flatten()
        grad_p_no_flat = torch.autograd.grad(p_loss.mean(),P_net.parameters())#.flatten()
        grad_p = grad_p_no_flat[0].flatten()
        beta=(grad_q*grad_p).sum()/(grad_q.norm(2,dim=0)*grad_p.norm(2,dim=0)+1e-12)
        # print(epoch,'beta=',beta.item(), )
        Beta.append(beta.item())
        Ss, As, Rs = get_batch(data, CHI_BATCH, True)
        q_loss = get_q_grad_loss(Ss,As,Q_net)
        p_loss = get_logp_grad_loss(Ss,As,partial(get_logp, p_policy=P_net))
        grad_q = torch.stack([torch.autograd.grad(qq_loss, Q_net.parameters(),retain_graph=True)[0] for qq_loss in q_loss]).flatten(1,2)
        grad_p = torch.stack([torch.autograd.grad(pp_loss, P_net.parameters(),retain_graph=True)[0] for pp_loss in p_loss]).flatten(1,2) #torch.autograd.grad(p_loss, P_net.parameters())
        chi= nn.CosineSimilarity(dim=1, eps=1e-6)(grad_q, grad_p) #(grad_q*grad_p).sum(dim=1)/(grad_q.norm(2,dim=1)*grad_p.norm(2,dim=1)+1e-12)
        # print(epoch,'chi=',chi.mean().item(), )
        Chi.append(chi.mean().item())
        lr = sch.get_last_lr()[0]

        for a,b in zip(grad_q_no_flat, Q_net.parameters()):
            b.data.add_(-lr*a)
        for a,b in zip(grad_p_no_flat, P_net.parameters()):
            b.data.add_(-lr*a)
        sch.step()
    return R, Beta, Chi
        # print('', )

@ray.remote
def casa_exp(seed, out_file=None):
    randomize(seed)

    env = GridWorld(num_rows=num_rows,
                num_cols=num_cols)

    Q_net = get_model(num_rows, num_cols)
    P_net = get_model(num_rows, num_cols)
    Casa_net = get_model(num_rows, num_cols)
    Casa_net.load_state_dict(P_net.state_dict()) # align casa and policy
    del Q_net
    del P_net
    Casa_v = get_model_v(num_rows, num_cols)
    optim = torch.optim.SGD(Casa_net.parameters(),lr=LR,weight_decay=0.0,momentum=0.9)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=1e-6, last_epoch=-1, verbose=False)
    R = []
    Beta = []
    Chi = []
    def _get_casa_q(ss):
        q,p = casa_q_pi(Casa_net(ss),Casa_v(ss))
        return q
    
    for epoch in range(EPOCHS):
        data, rs = get_data(env, Casa_net, MCSAMPLES)
        # print(epoch,'r=',rs,  )
        R.append(rs)
        Ss,As,Rs = get_batch(data,len(data),False)
        
        q_loss = get_Q_loss(Ss,As,Rs, _get_casa_q)
        p_loss = get_J_loss(Ss,As,Rs, partial(get_logp, p_policy=Casa_net))
        v_loss = get_V_loss(Ss,Rs, Casa_v)
        # print(q_loss.shape,p_loss.shape)
        # print(q_loss.mean().item(),p_loss.mean().item())
        grad_q_no_flat = torch.autograd.grad(q_loss.mean(),Casa_net.parameters())
        grad_q = grad_q_no_flat[0].flatten()
        grad_p_no_flat = torch.autograd.grad(p_loss.mean(),Casa_net.parameters())#.flatten()
        grad_p = grad_p_no_flat[0].flatten()

        grad_v = torch.autograd.grad(v_loss.mean(),Casa_v.parameters())

        beta=(grad_q*grad_p).sum()/(grad_q.norm(2,dim=0)*grad_p.norm(2,dim=0)+1e-12)
        # print(epoch,'beta=',beta.item(), )
        Beta.append(beta.item())
        Ss, As, Rs = get_batch(data, CHI_BATCH, True)
        q_loss = get_q_grad_loss(Ss,As,_get_casa_q)
        p_loss = get_logp_grad_loss(Ss,As,partial(get_logp, p_policy=Casa_net))
        grad_q = torch.stack([torch.autograd.grad(qq_loss, Casa_net.parameters(),retain_graph=True)[0] for qq_loss in q_loss]).flatten(1,2)
        grad_p = torch.stack([torch.autograd.grad(pp_loss, Casa_net.parameters(),retain_graph=True)[0] for pp_loss in p_loss]).flatten(1,2) #torch.autograd.grad(p_loss, P_net.parameters())
        # chi=(grad_q*grad_p).sum(dim=1)/(grad_q.norm(2,dim=1)*grad_p.norm(2,dim=1)+1e-12)
        chi = nn.CosineSimilarity(dim=1, eps=1e-6)(grad_q, grad_p)
        # print(epoch,'chi=',chi.mean().item(), )
        Chi.append(chi.mean().item())
        lr = sch.get_last_lr()[0]

        for a,b,c in zip(grad_q_no_flat,grad_p_no_flat, Casa_net.parameters()):
            c.data.add_(-lr/2.0*(a+b))
        for a,b in zip(grad_v, Casa_v.parameters()):
            b.data.add_(-lr*a)
        sch.step()
    return R, Beta, Chi
        # print('', )

@ray.remote
def casa_mix_exp(seed, alpha=0.5):
    randomize(seed)

    env = GridWorld(num_rows=num_rows,
                num_cols=num_cols)

    Q_net = get_model(num_rows, num_cols)
    P_net = get_model(num_rows, num_cols)
    Casa_net = get_model(num_rows, num_cols)
    Casa_net.load_state_dict(P_net.state_dict()) # align casa and policy
    del Q_net
    # del P_net
    Casa_v = get_model_v(num_rows, num_cols)
    optim = torch.optim.SGD(Casa_net.parameters(),lr=LR,weight_decay=0.0,momentum=0.9)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=1e-6, last_epoch=-1, verbose=False)
    R = []
    Beta = []
    Beta1 = []
    Chi = []
    Chi1 = []
    def _get_casa_q(ss):
        a = Casa_net(ss)
        v = Casa_v(ss)
        logits = P_net(ss)
        q,p = casa_mix_q_pi(a, v, logits, alpha, tau=1.0)
        return q
    def _get_log_p(ss):
        a = Casa_net(ss)
        v = Casa_v(ss)
        logits = P_net(ss)
        q,p = casa_mix_q_pi(a, v, logits, alpha, tau=1.0)
        return p.log()
    def _get_p(ss):
        a = Casa_net(ss)
        v = Casa_v(ss)
        logits = P_net(ss)
        q,p = casa_mix_q_pi(a, v, logits, alpha, tau=1.0)
        return p
    for epoch in range(EPOCHS):
        data, rs = get_data(env, _get_p, MCSAMPLES)
        # print(epoch,'r=',rs,  )
        R.append(rs)
        Ss,As,Rs = get_batch(data,len(data),False)
        
        q_loss = get_Q_loss(Ss,As,Rs, _get_casa_q)
        p_loss = get_J_loss(Ss,As,Rs, _get_log_p)
        v_loss = get_V_loss(Ss,Rs, Casa_v)
        # print(q_loss.shape,p_loss.shape)
        # print(q_loss.mean().item(),p_loss.mean().item())
        grad_q_no_flat = torch.autograd.grad(q_loss.mean(),Casa_net.parameters(),retain_graph=True)
        grad_q = grad_q_no_flat[0].flatten()
        grad_p_no_flat = torch.autograd.grad(p_loss.mean(),Casa_net.parameters(),retain_graph=True)#.flatten()
        grad_p = grad_p_no_flat[0].flatten()

        grad_v = torch.autograd.grad(v_loss.mean(),Casa_v.parameters(),retain_graph=True)

        beta=(grad_q*grad_p).sum()/(grad_q.norm(2,dim=0)*grad_p.norm(2,dim=0)+1e-12)
        # print(epoch,'beta=',beta.item(), )
        Beta.append(beta.item())
        grad_p_no_flat1 = torch.autograd.grad(p_loss.mean(),P_net.parameters(),retain_graph=True)#.flatten()
        grad_p = grad_p_no_flat1[0].flatten()
        beta=(grad_q*grad_p).sum()/(grad_q.norm(2,dim=0)*grad_p.norm(2,dim=0)+1e-12)
        # print(epoch,'beta=',beta.item(), )
        Beta1.append(beta.item())


        Ss, As, Rs = get_batch(data, CHI_BATCH, True)
        q_loss = get_q_grad_loss(Ss,As,_get_casa_q)
        p_loss = get_logp_grad_loss(Ss,As,_get_log_p)
        grad_q = torch.stack([torch.autograd.grad(qq_loss, Casa_net.parameters(),retain_graph=True)[0] for qq_loss in q_loss]).flatten(1,2)
        grad_p = torch.stack([torch.autograd.grad(pp_loss, Casa_net.parameters(),retain_graph=True)[0] for pp_loss in p_loss]).flatten(1,2) #torch.autograd.grad(p_loss, P_net.parameters())
        # chi=(grad_q*grad_p).sum(dim=1)/(grad_q.norm(2,dim=1)*grad_p.norm(2,dim=1)+1e-12)
        chi = nn.CosineSimilarity(dim=1, eps=1e-6)(grad_q, grad_p)
        
        # print(epoch,'chi=',chi.mean().item(), )
        Chi.append(chi.mean().item())

        grad_p = torch.stack([torch.autograd.grad(pp_loss, P_net.parameters(),retain_graph=True)[0] for pp_loss in p_loss]).flatten(1,2) #torch.autograd.grad(p_loss, P_net.parameters())
        chi = nn.CosineSimilarity(dim=1, eps=1e-6)(grad_q, grad_p)
        Chi1.append(chi.mean().item())
        lr = sch.get_last_lr()[0]

        for a,b,c in zip(grad_q_no_flat,grad_p_no_flat, Casa_net.parameters()):
            c.data.add_(-lr/2.0*(a+b))
        for a,c in zip(grad_p_no_flat1, P_net.parameters()):
            c.data.add_(-lr*a)
        for a,b in zip(grad_v, Casa_v.parameters()):
            b.data.add_(-lr*a)
        sch.step()
    return R, Beta, Chi, Beta1, Chi1

@ray.remote
def mpo_exp(seed, alpha):
    randomize(seed)

    env = GridWorld(num_rows=num_rows,
                num_cols=num_cols)

    Q_net = get_model(num_rows, num_cols)
    P_net = get_model(num_rows, num_cols)
    # Casa_net = get_model(num_rows, num_cols)
    # Casa_net.load_state_dict(P_net.state_dict()) # align casa and policy
    # del Q_net
    # del P_net
    Casa_v = get_model_v(num_rows, num_cols)
    optim = torch.optim.SGD(list(Q_net.parameters())+list(P_net.parameters()),lr=LR,weight_decay=0.0,momentum=0.9)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=1e-6, last_epoch=-1, verbose=False)
    R = []
    Beta = []
    Chi = []
    def _get_log_p(ss):
        q = Q_net(ss)
        logits = P_net(ss)
        log_p = log_mpo_pi(q, logits, alpha=alpha, tau=1.0)
        # q,p = casa_q_pi(Casa_net(ss),Casa_v(ss))
        return log_p
    def _get_p(ss):
        return _get_log_p(ss).exp()
    for epoch in range(EPOCHS):
        data, rs = get_data(env, _get_p, MCSAMPLES)
        # print(epoch,'r=',rs,  )
        R.append(rs)
        Ss,As,Rs = get_batch(data,len(data),False)
        
        q_loss = get_Q_loss(Ss,As,Rs, Q_net)
        p_loss = get_J_loss(Ss,As,Rs, _get_log_p)

        grad_q_no_flat = torch.autograd.grad(q_loss.mean(),Q_net.parameters())
        grad_q = grad_q_no_flat[0].flatten()
        grad_p_no_flat = torch.autograd.grad(p_loss.mean(),P_net.parameters())#.flatten()
        grad_p = grad_p_no_flat[0].flatten()
        beta=(grad_q*grad_p).sum()/(grad_q.norm(2,dim=0)*grad_p.norm(2,dim=0)+1e-12)
        # print(epoch,'beta=',beta.item(), )
        Beta.append(beta.item())
        Ss, As, Rs = get_batch(data, CHI_BATCH, True)
        q_loss = get_q_grad_loss(Ss,As, Q_net)
        p_loss = get_logp_grad_loss(Ss,As, _get_log_p)
        grad_q = torch.stack([torch.autograd.grad(qq_loss, Q_net.parameters(),retain_graph=True)[0] for qq_loss in q_loss]).flatten(1,2)
        grad_p = torch.stack([torch.autograd.grad(pp_loss, P_net.parameters(),retain_graph=True)[0] for pp_loss in p_loss]).flatten(1,2) #torch.autograd.grad(p_loss, P_net.parameters())
        # chi=(grad_q*grad_p).sum(dim=1)/(grad_q.norm(2,dim=1)*grad_p.norm(2,dim=1)+1e-12)
        chi = nn.CosineSimilarity(dim=1, eps=1e-6)(grad_q, grad_p)
        # print(epoch,'chi=',chi.mean().item(), )
        Chi.append(chi.mean().item())
        lr = sch.get_last_lr()[0]

        for a,b in zip(grad_q_no_flat, Q_net.parameters()):
            b.data.add_(-lr*a)
        for a,b in zip(grad_p_no_flat, P_net.parameters()):
            b.data.add_(-lr*a)

        sch.step()
    return R, Beta, Chi

from multiprocessing import  Process
if __name__=='__main__':
    import ray

    ray.init()

    
    f1s = []
    f2s = []
    f3s = []
    Casa_results=[]

    for i in range(5):
        for j in range(5):
        # f1 = f'../results/baseline_exp_44fast_{i}.txt'
        # f2 = f'../results/casa_exp_44fast_{i}.txt'
            f3 = f'../results/casa_mix_alpha_{0.2*i}_{j}.txt'
            Casa_results.append(casa_mix_exp.remote(2021+j,0.2*i))
        # f1 = open(f1,'w')
        # f2 = open(f2,'w')
        # f1s.append(f1)
        # f2s.append(f2)
            f3s.append(f3)
    process_list = []
    Baseline_results = []
    # Casa_results = []
    # for i,(f1,f2) in enumerate(zip(f1s,f2s)):
    #     # print(i,f1,f2)
    #     # Baseline_results.append(baseline_exp.remote(2021))
    #     # Casa_results.append(casa_exp.remote(2021+i))
    #     Mpo_results.append(mpo_exp.remote(2021+i))
    #     # p = Process(target=baseline_exp,args=(2020+i, f2)) 
    #     # p.start()
    #     # process_list.append(p)
    #     # p = Process(target=casa_exp,args=(2020+i, f1)) 
    #     # p.start()
    #     # process_list.append(p)
    # Baseline_results = ray.get(Baseline_results)
    Casa_results = ray.get(Casa_results)
    # Mpo_results = ray.get(Mpo_results)
    for (mpo,f3) in zip(Casa_results,f3s):
        # with open(f1,'w') as f:
        #     R,B,C = base
        #     for r,b,c in zip(R,B,C):
        #         f.write(f'{r},{b},{c}\n')
        # with open(f2,'w') as f:
        #     R,B,C = casa
        #     for r,b,c in zip(R,B,C):
        #         f.write(f'{r},{b},{c}\n')
        with open(f3,'w') as f:
            R,B,C,B1,C1 = mpo
            for r,b,c,b1,c1 in zip(R,B,C,B1,C1):
                f.write(f'{r},{b},{c},{b1},{c1}\n')
    ray.shutdown()
    assert 0
    # with open('casa_exp0.txt','w') as f:
        # casa_exp(2022,f)

    # print(q_loss.shape,p_loss.shape,grad_q.shape,grad_p.shape)
