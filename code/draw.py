import pickle

import os
import matplotlib.pyplot as plt
import numpy as np

def get_file_names(path):
    ans = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            n = os.path.join(root,name)
            if '_mix_alpha_' in n:
                ans.append(os.path.join(root, name))
                print("'"+os.path.join(root, name)+"',")
    return ans

paths = get_file_names('./')
# assert 0
def get_average(x,alpha=50.0):
    import numpy as np
    v = x[0]
    v_new = [v]
    if  alpha > 1:
        alpha = int(alpha)
    for i in range(1,len(x)):
        v = alpha*x[i]+(1-alpha)*v
        if alpha >1:
            v_new.append(np.mean(x[max(0,i-alpha):i]))
        else:
            v_new.append(v)
    return v_new

def handle_one_plot(x, cut=0,total_f=200):
    xx=[]
    yy=[]
    for i in x:
        (a,b) = i
        xx.append(a)
        yy.append(b)
    total = len(xx)
    xx = [total_f*(1e6)/total*(i+1) for i in range(len(xx))]
    new_x = []
    new_y = []
    for xxx,yyy in zip(xx,yy):
        if xxx>=cut*(1e6):
            new_x.append(xxx)
            new_y.append(yyy)
    return new_x,new_y
# assert 0

# paths = ['./casa_exp_44fast_4.txt',
# './casa_exp_44fast_0.txt',
# './baseline_exp_44fast_3.txt',
# './baseline_exp_44fast_2.txt',
# './baseline_exp_44fast_8.txt',
# './casa_exp_44fast_7.txt',
# './casa_exp_44fast_9.txt',
# './baseline_exp_44fast_0.txt',
# './baseline_exp_44fast_6.txt',
# './casa_exp_44fast_1.txt',
# './casa_exp_44fast_3.txt',
# './baseline_exp_44fast_5.txt',
# './casa_exp_44fast_2.txt',
# './casa_exp_44fast_6.txt',
# './casa_exp_44fast_5.txt',
# './casa_exp_44fast_8.txt',
# './baseline_exp_44fast_9.txt',
# './baseline_exp_44fast_1.txt',
# './baseline_exp_44fast_7.txt',
# './baseline_exp_44fast_4.txt',
# ]

def handle_file(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
        rs=[0.0 for _ in range(1000)]
        bs=[0.0 for _ in range(1000)]
        cs=[0.0 for _ in range(1000)]
        b1s=[0.0 for _ in range(1000)]
        c1s=[0.0 for _ in range(1000)]
        for epoch, line in enumerate(lines):
            r,b,c,b1,c1 = line.split(',')
            rs[epoch] = float(r)
            bs[epoch] = float(b)
            cs[epoch] = float(c)
            b1s[epoch] = float(b1)
            c1s[epoch] = float(c1)
    f=[rs,bs,cs,b1s,c1s]
    f = np.array(f)
    return f
        
# casa = [handle_file(f) for f in paths if 'baseline' in f]
# baseline = [handle_file(f) for f in paths if 'casa' in f]
def moving_average(x):
    d = [0.0 for _ in x]
    d[0] = x[0]
    for i in range(1, len(x)):
        d[i] = d[i-1]*0.95+0.05*x[i]
    return d
mpo = [[handle_file(f) for f in paths if f'0.{2*i}' in f] for i in range(5)]

    # print(i.shape)
mpo = np.stack([np.stack(i) for i in mpo])
print(mpo.shape)
# assert 0
# print(casa,baseline)
# print(len(mpo),len(casa),len(baseline))
mpo_m=[]
mpo_std=[]
# for m in mpos:
#     m = [[np.mean([casa[i][j][k] for i in range(10)]) for k in range(1000)] for j in range(3)]
#     std = [[np.std([casa[i][j][k] for i in range(10)]) for k in range(1000)] for j in range(3)]

# base_m = [[np.mean([baseline[i][j][k] for i in range(10)]) for k in range(1000)] for j in range(3)]
# base_std = [[np.std([baseline[i][j][k] for i in range(10)]) for k in range(1000)] for j in range(3)]

# mpo_m = [[np.mean([mpo[i][j][k] for i in range(10)]) for k in range(1000)] for j in range(3)]
# mpo_std = [[np.std([mpo[i][j][k] for i in range(10)]) for k in range(1000)] for j in range(3)]
# print(casa_r,casa_std)
plt.figure(figsize = (24,11))
plt.subplot(3,1,1)
x = list(range(1000))
for i in range(5):
    y = np.mean(mpo[i],axis=0,keepdims=False)
    plt.plot(x,moving_average(y[0]),c=f'C{i}', label=f'casa_mix_0.{2*i}')
# plt.fill_between(x,[i+j for i,j in zip(casa_m[0],casa_std[0])],[i-j for i,j in zip(casa_m[0],casa_std[0])],color='C0',alpha=0.5)
# plt.plot(x,base_m[0],c='C1', label='baseline')
# plt.fill_between(x,[i+j for i,j in zip(base_m[0],base_std[0])],[i-j for i,j in zip(base_m[0],base_std[0])],color='C1',alpha=0.5)
# plt.plot(x,mpo_m[0],c='C2', label='mpo')
# plt.fill_between(x,[i+j for i,j in zip(mpo_m[0],mpo_std[0])],[i-j for i,j in zip(mpo_m[0],mpo_std[0])],color='C2',alpha=0.5)
plt.legend()


plt.subplot(3,1,2)
for i in range(5):
    y = np.mean(mpo[i],axis=0,keepdims=False)
    plt.plot(x,moving_average(y[1]),c=f'C{i}', label=f'casa_mix_0.{2*i}')
# plt.plot(x,casa_m[1],c='C0')
# plt.fill_between(x,[i+j for i,j in zip(casa_m[1],casa_std[1])],[i-j for i,j in zip(casa_m[1],casa_std[1])],color='C0',alpha=0.5)

# plt.plot(x,base_m[1],c='C1')
# plt.fill_between(x,[i+j for i,j in zip(base_m[1],base_std[1])],[i-j for i,j in zip(base_m[1],base_std[1])],color='C1',alpha=0.5)

# plt.plot(x,mpo_m[1],c='C2')
# plt.fill_between(x,[i+j for i,j in zip(mpo_m[1],mpo_std[1])],[i-j for i,j in zip(mpo_m[1],mpo_std[1])],color='C2',alpha=0.5)
plt.ylim(0,1.05)

# plt.subplot(5,1,3)
# for i in range(5):
    # y = np.mean(mpo[i],axis=0,keepdims=False)
    # plt.plot(x,moving_average(y[2]),c=f'C{i}', label=f'casa_mix_0.{2*i}')
# plt.plot(x,casa_m[2],c='C0')
# plt.fill_between(x,[i+j for i,j in zip(casa_m[2],casa_std[2])],[i-j for i,j in zip(casa_m[2],casa_std[2])],color='C0',alpha=0.5)
# plt.plot(x,base_m[2],c='C1')
# plt.fill_between(x,[i+j for i,j in zip(base_m[2],base_std[2])],[i-j for i,j in zip(base_m[2],base_std[2])],color='C1',alpha=0.5)
# plt.plot(x,mpo_m[2],c='C2')
# plt.fill_between(x,[i+j for i,j in zip(mpo_m[2],mpo_std[2])],[i-j for i,j in zip(mpo_m[2],mpo_std[2])],color='C2',alpha=0.5)
plt.ylim(0,1.05)
plt.subplot(3,1,3)
for i in range(5):
    y = np.mean(mpo[i],axis=0,keepdims=False)
    plt.plot(x,moving_average(y[3]),c=f'C{i}', label=f'casa_mix_0.{2*i}')
# plt.plot(x,casa_m[2],c='C0')
# plt.fill_between(x,[i+j for i,j in zip(casa_m[2],casa_std[2])],[i-j for i,j in zip(casa_m[2],casa_std[2])],color='C0',alpha=0.5)
# plt.plot(x,base_m[2],c='C1')
# plt.fill_between(x,[i+j for i,j in zip(base_m[2],base_std[2])],[i-j for i,j in zip(base_m[2],base_std[2])],color='C1',alpha=0.5)
# plt.plot(x,mpo_m[2],c='C2')
# plt.fill_between(x,[i+j for i,j in zip(mpo_m[2],mpo_std[2])],[i-j for i,j in zip(mpo_m[2],mpo_std[2])],color='C2',alpha=0.5)
# plt.ylim(0,1.05)
# plt.subplot(5,1,5)
# for i in range(5):
    # y = np.mean(mpo[i],axis=0,keepdims=False)
    # plt.plot(x,moving_average(y[4]),c=f'C{i}', label=f'casa_mix_0.{2*i}')
# plt.plot(x,casa_m[2],c='C0')
# plt.fill_between(x,[i+j for i,j in zip(casa_m[2],casa_std[2])],[i-j for i,j in zip(casa_m[2],casa_std[2])],color='C0',alpha=0.5)
# plt.plot(x,base_m[2],c='C1')
# plt.fill_between(x,[i+j for i,j in zip(base_m[2],base_std[2])],[i-j for i,j in zip(base_m[2],base_std[2])],color='C1',alpha=0.5)
# plt.plot(x,mpo_m[2],c='C2')
# plt.fill_between(x,[i+j for i,j in zip(mpo_m[2],mpo_std[2])],[i-j for i,j in zip(mpo_m[2],mpo_std[2])],color='C2',alpha=0.5)
# plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig('../results/test_mix.jpg')

assert 0



def draw_new_ablation(game_name):
    # plt.figure(4)

    plt.figure(figsize=(36,5))
    dds=get_various_dict(paths, game_name)
    ab_names = ['newab0','newab1','newab2','newab3','newab4','newab5']
    sns = ['PPO+CASA','type 1','type 2','type 3','type 4','type 5']
    cs = [ '#e8710a','#7cb342','#e80a0a','#425066','#12b5cb','#e52592']
    plt.subplot(1,5,1)

    for abn,sn,c in zip(ab_names,sns,cs):
        l0 = handle_one_plot(dds[abn]['return_avg'],cut=0,total_f=50)
        l0_min = handle_one_plot(dds[abn]['return_min'],cut=0,total_f=50)
        l0_max = handle_one_plot(dds[abn]['return_max'],cut=0,total_f=50)
        tmp_ave = lambda x:get_average(x[1],150)
        plt.plot(l0[0],get_average(l0[1],150),c=c,label=sn, alpha=1.0, linewidth=4.0)
        plt.fill_between(l0_min[0],tmp_ave(l0_min),tmp_ave(l0_max),facecolor=c,alpha=0.5)    
    plt.grid(b=True, which='major', axis='both', )
    plt.xticks([0.5/8*i*(10**8) for i in range(9)],[str(int(50/8*i))+'' for i in range(9)],fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left',fontsize=18)
    plt.xlabel('Millions of frames',fontsize=28)
    plt.ylabel(game_name+' Return',fontsize=28)
    def plot_scatter(plt, dds, i):
        plt.subplot(1, 5, i)
        kk = ['newab0','newab1','newab2','newab3','newab4','newab5']
        sns = ['PPO+CASA','type 1','type 2','type 3','type 4','type 5']
        cs = [ '#e8710a','#7cb342','#e80a0a','#425066','#12b5cb','#e52592']
        ls = [handle_one_plot(dds[a]['chi_logp_q_cos'],cut=0,total_f=50) for a in kk]
        rs = [handle_one_plot(dds[a]['grad_pg2_q_cos'],cut=0,total_f=50) for a in kk]
        for l,r,c,sn in zip(ls,rs, cs,sns):
            plt.scatter(l[1][::50],r[1][::50],c=c,label=sn,alpha=0.7)
        plt.grid(b=True, which='major', axis='both', )
    def plot_cos(plt, dds, i, key):
        plt.subplot(1, 5, i)
        kk = ['newab0','newab1','newab2','newab3','newab4','newab5']
        sns = ['PPO+CASA','type 1','type 2','type 3','type 4','type 5']
        cs = [ '#e8710a','#7cb342','#e80a0a','#425066','#12b5cb','#e52592']
        ls = [handle_one_plot(dds[a][key],cut=0,total_f=50) for a in kk]
        # cs = ['#9334e6','#f9ab00','#12b5cb','#e52592']
        for l,c,sn in zip(ls,cs,sns):
            # if k =='ppo' and 'q' in key:
                # continue
            plt.plot(l[0],get_average(l[1],150),c=c,label=sn, alpha=1.0, linewidth=4.0)
            
        plt.grid(b=True, which='major', axis='both', )
        plt.xticks([0.5/8*i*(10**8) for i in range(1,9)],[str(int(50/8*i))+'' for i in range(1,9)],fontsize=18)
        plt.yticks(fontsize=18)
        # plt.xlabel('Millions of frames',fontsize=28)
        # plt.ylabel('cos $<\\nabla Q, \\nabla J>$',fontsize=28)
    def plot_box(plt, dds, i):
        plt.subplot(1, 5, i)
        kk = ['newab0','newab1','newab3','newab4','newab5','newab2',]
        sns = ['PPO+CASA','type 1','type 3','type 4','type 5','type 2',]
        cs = [ '#e8710a','#7cb342','#425066','#12b5cb','#e52592','#e80a0a',]
        ls = [handle_one_plot(dds[a]['grad_pg2_q_cos'],cut=0,total_f=50) for a in kk]
        rs = [handle_one_plot(dds[a]['chi_logp_q_cos'],cut=0,total_f=50) for a in kk]
        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        boxp = plt.boxplot([l[1][::10] for l in ls], positions=[np.mean(r[1]) for r in rs], labels=sns, sym='',notch=True,
            patch_artist=True, widths=0.075, manage_ticks=False)
        for box,c in zip(boxp['boxes'],cs):
            box.set_facecolor(c)
        plt.grid(b=True, which='major', axis='both', )
        # plt.legend([boxp['boxes'][i] for i in [0,1,5,2,3,4]],[sns[i] for i in [0,1,5,2,3,4]],loc='upper left')
        # plt.xticks([i*(1.3+0.5)/10-0.5 for i in range(1,10)],[str(i*(1.2+0.2)/10-0.5)[:4] for i in range(1,10)],
        # fontsize=14)
        # plt.ylabel('cos $<\\nabla L_Q, \\nabla J>$',fontsize=28)
        plt.xlim(-0.5,1.1)
        plt.ylim(-0.5,1.0)
        plt.ylabel('cos $<\\nabla L_Q, \\nabla J>$',fontsize=28)
        plt.xlabel('$\chi$',fontsize=28)
    
    plot_cos(plt,dds,2,'chi_logp_q_cos')
    plt.ylabel('$\chi$',fontsize=28)
    plt.xlabel('Millions of frames',fontsize=28)
    plt.ylim(-0.1,1.05)
    # plot_cos(plt,dds,3,'grad_q_v_cos')
    # plt.ylabel('cos $<\\nabla L_Q, \\nabla L_V>$',fontsize=28)

    # plt.ylim(-0.1,1.05)
    # plot_cos(plt,dds,4,'grad_pg2_v_cos')
    # plt.ylabel('cos $<\\nabla J, \\nabla L_v>$',fontsize=28)

    # plt.ylim(-0.1,1.05)
    plot_cos(plt,dds,3,'grad_pg2_q_cos')
    plt.xlabel('Millions of frames',fontsize=28)
    plt.ylabel('cos $<\\nabla L_Q, \\nabla J>$',fontsize=28)

    plt.ylim(-0.1,1.05)

    plot_scatter(plt,dds, 4)
    plt.xlabel('$\chi$',fontsize=28)
    plt.ylabel('cos $<\\nabla L_Q, \\nabla J>$',fontsize=28)

    plt.xlim(-0.5,1.0)
    plt.ylim(-0.5,1.0)
    plot_box(plt, dds, 5)
    
    plt.tight_layout()
    plt.savefig(f'new_ab_{game_name}.pdf')


    # draw_drtrace_abla()