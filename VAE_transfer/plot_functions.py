import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import tikzplotlib
import csv

from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

def plot_souce_target(source_data, target_data):
    fig, ax = plt.subplots(1,3, figsize=(16,4),gridspec_kw={"width_ratios":[1,1,1]})

    if isinstance(source_data, pd.DataFrame):
        source_z = source_data['z']
    else:  
        source_data = source_data.dataset_to_dataframe()
        source_z = source_data['z0']
    if isinstance(target_data, pd.DataFrame):
        target_z = target_data['z']
    else:
        target_data = target_data.dataset_to_dataframe()
        target_z = target_data['z0']  

    ax[0].hist(source_z, bins=30, alpha=0.5, label='Source', density=True)
    ax[0].hist(target_z, bins=30, alpha=0.5, label='Target', density=True)
    ax[0].set_title("Distribution of $Z$", fontsize=18)
    ax[0].set_xlabel('z', fontsize=16)
    ax[0].set_ylabel("Density", fontsize=16)
    ax[0].tick_params(which='major', axis='both', labelsize=16)
    ax[0].legend()

    ax[1].hist(source_data.loc[source_data['t']==0, 'y'], alpha = 0.5, label = 't=0', bins=30, density=True)
    ax[1].hist(source_data.loc[source_data['t']==1, 'y'], alpha=0.5, label='t=1', bins=30, density=True)
    ax[1].set_title("$y$ distribution in source domain", fontsize=18)
    ax[1].set_xlabel('$y$', fontsize=16)
    ax[1].set_ylabel("Density", fontsize=16)
    ax[1].tick_params(which='major', axis='both', labelsize=16)
    ax[1].legend()

    ax[2].hist(target_data.loc[target_data['t']==0, 'y'], alpha = 0.5, label = 't=0', bins=30, density=True)
    ax[2].hist(target_data.loc[target_data['t']==1, 'y'], alpha=0.5, label='t=1', bins=30, density=True)
    ax[2].set_title("$y$ distribution in target domain", fontsize=18)
    ax[2].set_xlabel('$y$', fontsize=16)
    ax[2].set_ylabel("Density", fontsize=16)
    ax[2].tick_params(which='major', axis='both', labelsize=16)
    ax[2].legend()

    plt.legend()
    plt.show()
    plt.savefig('results/IHDP_dist.png', dpi=300)
    #tikzplotlib.save("results/IHDP_dist.tex")

def plot_py_xt(source_data, target_data, py_xt, num_test_sample, loss_type=1):
    
    x_source = np.array(source_data.iloc[:, 1:25].sample(num_test_sample, random_state=10))
    x_target = np.array(target_data.iloc[:, 1:25].sample(num_test_sample, random_state=10))


    t0 = np.zeros(len(x_source)).reshape(-1, 1)
    t1 = np.ones(len(x_source)).reshape(-1, 1)  
    xt0 = torch.Tensor(np.concatenate([x_source, t0], 1))
    xt1 = torch.Tensor(np.concatenate([x_source, t1], 1))
    
    if loss_type == 1:
        y0_s = py_xt(xt0).detach().numpy()
        y1_s = py_xt(xt1).detach().numpy()
    else:
        y0_s_dist = py_xt(xt0)
        y1_s_dist = py_xt(xt1)

        y0_s_mu = y0_s_dist[:, :1]
        y0_s_std = torch.exp(y0_s_dist[:, 1:])
        y1_s_mu = y1_s_dist[:, :1]
        y1_s_std = torch.exp(y1_s_dist[:, 1:])

        y0_s = dist.Normal(loc=y0_s_mu, scale=y0_s_std).sample().detach().numpy()
        y1_s =dist.Normal(loc=y1_s_mu, scale=y1_s_std).sample().detach().numpy()

    t0 = np.zeros(len(x_target)).reshape(-1, 1)
    t1 = np.ones(len(x_target)).reshape(-1, 1)
    xt0 = torch.Tensor(np.concatenate([x_target, t0], 1))
    xt1 = torch.Tensor(np.concatenate([x_target, t1], 1))

    if loss_type == 1:
        y0_t = py_xt(xt0).detach().numpy()
        y1_t = py_xt(xt1).detach().numpy()
    else:
        y0_t_dist = py_xt(xt0)
        y1_t_dist = py_xt(xt1)

        y0_t_mu = y0_t_dist[:, :1]
        y0_t_std = torch.exp(y0_t_dist[:, 1:])
        y1_t_mu = y1_t_dist[:, :1]
        y1_t_std = torch.exp(y1_t_dist[:, 1:])

        y0_t = dist.Normal(loc=y0_t_mu, scale=y0_t_std).sample().detach().numpy()
        y1_t =dist.Normal(loc=y1_t_mu, scale=y1_t_std).sample().detach().numpy()




    fig, ax = plt.subplots(1,2, figsize=(16,4),gridspec_kw={"width_ratios":[1,1]})
    ax[0].hist(y0_s, alpha = 0.5, label = 't=0', bins=20)
    ax[0].hist(y1_s, alpha=0.5, label='t=1', bins=20)
    ax[0].set_title("$p(y|x,t)$  in source domain", fontsize=18)
    ax[0].set_xlabel('$y$', fontsize=16)
    ax[0].set_ylabel("frequency", fontsize=16)
    ax[0].tick_params(which='major', axis='both', labelsize=16)
    ax[0].legend()


    ax[1].hist(y0_t, alpha = 0.5, label = 't=0', bins=20)
    ax[1].hist(y1_t, alpha=0.5, label='t=1', bins=20)
    ax[1].set_title("$p(y|x,t)$  in target domain", fontsize=18)
    ax[1].set_xlabel('$y$', fontsize=16)
    ax[1].set_ylabel("frequency", fontsize=16)
    ax[1].tick_params(which='major', axis='both', labelsize=16)
    ax[1].legend()

    plt.show()



def plot_py_zt(source_data, target_data, py_zt, num_test_sample, loss_type=1, density=True):

    z_source = []
    z_target = []
    if 'z0' in source_data.columns:
        z_source = source_data['z0'].sample(num_test_sample, random_state=1)
        z_target = target_data['z0'].sample(num_test_sample, random_state=1)
    else:
        z_source = source_data['z'].sample(num_test_sample, random_state=1)
        z_target = target_data['z'].sample(num_test_sample, random_state=1)
    #z_target = np.random.normal(0, 1, size=num_test_sample)
    z_source = np.array(z_source).reshape(-1,1)
    z_target = np.array(z_target).reshape(-1,1)

    t0 = np.zeros(len(z_source)).reshape(-1, 1)
    zt0 = torch.Tensor(np.concatenate([z_source, t0], 1))
    t1 = np.ones(len(z_source)).reshape(-1, 1)
    zt1 = torch.Tensor(np.concatenate([z_source, t1], 1))
    if loss_type == 1:
        y0_s = py_zt(torch.Tensor(z_source), torch.Tensor(t0)).detach().numpy()
        y1_s = py_zt(torch.Tensor(z_source), torch.Tensor(t1)).detach().numpy()
    else:
        y0_s_mu, y0_s_std = py_zt(torch.Tensor(z_source), torch.Tensor(t0))
        y1_s_mu, y1_s_std = py_zt(torch.Tensor(z_source), torch.Tensor(t1))
        y0_s = dist.Normal(loc=y0_s_mu, scale=y0_s_std).sample().detach().numpy()
        y1_s =dist.Normal(loc=y1_s_mu, scale=y1_s_std).sample().detach().numpy()

    t0 = np.zeros(len(z_target)).reshape(-1, 1)
    t1 = np.ones(len(z_target)).reshape(-1, 1)
    zt0 = torch.Tensor(np.concatenate([z_target, t0], 1))
    zt1 = torch.Tensor(np.concatenate([z_target, t1], 1))

    if loss_type == 1:
        y0_t = py_zt(torch.Tensor(z_target), torch.Tensor(t0)).detach().numpy()
        y1_t = py_zt(torch.Tensor(z_target), torch.Tensor(t1)).detach().numpy()
    else:
        y0_t_mu, y0_t_std = py_zt(torch.Tensor(z_target), torch.Tensor(t0))
        y1_t_mu, y1_t_std = py_zt(torch.Tensor(z_target), torch.Tensor(t1))
        y0_t = dist.Normal(loc=y0_t_mu, scale=y0_t_std).sample().detach().numpy()
        y1_t =dist.Normal(loc=y1_t_mu, scale=y1_t_std).sample().detach().numpy()

    bins = 20
    fig, ax = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1]})

    # Plotting source domain histograms
    ax[0].hist(y0_s, alpha=0.5, label='t=0', bins=bins, density=density)
    ax[0].hist(y1_s, alpha=0.5, label='t=1', bins=bins, density=density)
    ax[0].set_title("$p(y|z,t)$ in source domain", fontsize=18)
    ax[0].set_xlabel('$y$', fontsize=16)    
    ax[0].set_ylabel("Density", fontsize=16)
    ax[0].tick_params(which='major', axis='both', labelsize=16)
    ax[0].legend()

    # Plotting target domain histograms
    ax[1].hist(y0_t, alpha=0.5, label='t=0', bins=bins, density=density)
    ax[1].hist(y1_t, alpha=0.5, label='t=1', bins=bins, density=density)
    ax[1].set_title("$p(y|z,t)$ in target domain", fontsize=18)
    ax[1].set_xlabel('$y$', fontsize=16)
    ax[1].set_ylabel("Density", fontsize=16)
    ax[1].tick_params(which='major', axis='both', labelsize=16)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_py_zt_image(source_dataframe, target_dataframe, device, model):
    density = True
    z_source = np.array([source_dataframe['z0'], source_dataframe['z1'], source_dataframe['z2']]).transpose()
    z_target = np.array([target_dataframe['z0'], target_dataframe['z1'], target_dataframe['z2']]).transpose()

    t0 = np.zeros(len(z_source)).reshape(-1, 1)
    zt0 = torch.Tensor(np.concatenate([z_source, t0], 1))
    t1 = np.ones(len(z_source)).reshape(-1, 1)
    zt1 = torch.Tensor(np.concatenate([z_source, t1], 1))

    _,_,_,_,_, y0_s_mu, y0_s_std = model(torch.Tensor(z_source).to(device), torch.Tensor(t0).to(device))
    _,_,_,_,_, y1_s_mu, y1_s_std = model(torch.Tensor(z_source).to(device), torch.Tensor(t1).to(device))
    y0_s = dist.Normal(loc=y0_s_mu, scale=y0_s_std).sample().detach().cpu().numpy()
    y1_s =dist.Normal(loc=y1_s_mu, scale=y1_s_std).sample().detach().cpu().numpy()


    t0 = np.zeros(len(z_target)).reshape(-1, 1)
    t1 = np.ones(len(z_target)).reshape(-1, 1)
    zt0 = torch.Tensor(np.concatenate([z_target, t0], 1))
    zt1 = torch.Tensor(np.concatenate([z_target, t1], 1))

    _,_,_,_,_, y0_t_mu, y0_t_std = model(torch.Tensor(z_target).to(device), torch.Tensor(t0).to(device))
    _,_,_,_,_, y1_t_mu, y1_t_std = model(torch.Tensor(z_target).to(device), torch.Tensor(t1).to(device))
    y0_t = dist.Normal(loc=y0_t_mu, scale=y0_t_std).sample().detach().cpu().numpy()
    y1_t =dist.Normal(loc=y1_t_mu, scale=y1_t_std).sample().detach().cpu().numpy()


    bins = 20
    fig, ax = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1]})

    # Plotting source domain histograms
    ax[0].hist(y0_s, alpha=0.5, label='t=0', bins=bins, density=density)
    ax[0].hist(y1_s, alpha=0.5, label='t=1', bins=bins, density=density)
    ax[0].set_title("$p(y|z,t)$ in source domain", fontsize=18)
    ax[0].set_xlabel('$y$', fontsize=16)    
    ax[0].set_ylabel("Density", fontsize=16)
    ax[0].tick_params(which='major', axis='both', labelsize=16)
    ax[0].legend()

    # Plotting target domain histograms
    ax[1].hist(y0_t, alpha=0.5, label='t=0', bins=bins, density=density)
    ax[1].hist(y1_t, alpha=0.5, label='t=1', bins=bins, density=density)
    ax[1].set_title("$p(y|z,t)$ in target domain", fontsize=18)
    ax[1].set_xlabel('$y$', fontsize=16)
    ax[1].set_ylabel("Density", fontsize=16)
    ax[1].tick_params(which='major', axis='both', labelsize=16)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_decoder(target_data, decoder, x_mode, x_dim, k):
    z = target_data['z']
    x = target_data.iloc[:, 1:x_dim+1] 
    n_x_estimands = sum([1 if m==0 or m==2 else m for m in x_mode])
    different_modes = list(set(x_mode))
    x_same_mode_indices = dict()
    for mode in different_modes:
        x_same_mode_indices[mode] = [i for i,m in enumerate(x_mode) if m==mode]

        z_sample = torch.FloatTensor(np.array(z).reshape(-1, 1))
        x_dist = decoder(z_sample)
        x_mu = x_dist[:,:n_x_estimands]
        x_std = torch.exp(x_dist[:,n_x_estimands:])      
        x_sample = np.zeros((len(z_sample), x_dim))

        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(x_mode):
            if mode==0:
                x_sample[:,i] = dist.Normal(loc=x_mu[:,pred_i], scale=x_std[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            elif mode==2:
                x_sample[:,i] = dist.Bernoulli(logits=x_mu[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            else:
                x_sample[:,i] = dist.Categorical(logits=x_mu[:,pred_i:pred_i+mode]).sample().detach().numpy()
                pred_i += mode

    plt.hist(x_sample[:,k],alpha=0.5, label='decoder', bins=30)
    plt.hist(x.iloc[:,k], alpha=0.5,label='p(x|z)', bins=30)
    plt.title(f'x{k}')
    plt.legend()

def plot_regret(regret_causal_transDe_episodes, regret_causal_NoDe_episodes, regret_causal_sourceVAE_episodes, regret_negative_episodes, regret_random_episodes, dist_z_source, dist_z_target):
    def compute_mean_and_variance(regret_episodes):
        if len(regret_episodes) == 0:
            mean_regret = np.array([np.nan])
            var_regret = np.array([np.nan])
        else:
            regret_array = np.array(regret_episodes)
            mean_regret = np.mean(regret_array, axis=0)
            var_regret = np.var(regret_array, axis=0)
        return mean_regret, var_regret

    # Compute mean and variance for each type of regret
    mean_regret_causal_transDe, var_regret_causal_transDe = compute_mean_and_variance(regret_causal_transDe_episodes)
    mean_regret_causal_NoDe, var_regret_causal_NoDe = compute_mean_and_variance(regret_causal_NoDe_episodes)
    mean_regret_causal_sourceVAE, var_regret_causal_sourceVAE = compute_mean_and_variance(regret_causal_sourceVAE_episodes)
    mean_regret_negative, var_regret_negative = compute_mean_and_variance(regret_negative_episodes)
    mean_regret_random, var_regret_random = compute_mean_and_variance(regret_random_episodes)

    # Plotting
    steps = np.arange(len(mean_regret_random))

    plt.figure(figsize=(8, 5))


    #plt.plot(steps, mean_regret_negative, label='Negative', color='red')
    #plt.fill_between(steps, mean_regret_negative - np.sqrt(var_regret_negative), mean_regret_negative + np.sqrt(var_regret_negative), alpha=0.2, color='red')
    if not np.isnan(mean_regret_causal_NoDe[0]):
        plt.plot(steps, mean_regret_causal_NoDe, label='Causal NoDe', color='orange')
        plt.fill_between(steps, mean_regret_causal_NoDe - np.sqrt(var_regret_causal_NoDe), mean_regret_causal_NoDe + np.sqrt(var_regret_causal_NoDe), alpha=0.2, color='orange')

    if not np.isnan(mean_regret_causal_sourceVAE[0]):
        plt.plot(steps, mean_regret_causal_sourceVAE, label='Causal SourceVAE', color='green')
        plt.fill_between(steps, mean_regret_causal_sourceVAE - np.sqrt(var_regret_causal_sourceVAE), mean_regret_causal_sourceVAE + np.sqrt(var_regret_causal_sourceVAE), alpha=0.2, color='green')


    if not np.isnan(mean_regret_random[0]):
        plt.plot(steps, mean_regret_random, label='Random', color='purple')
        plt.fill_between(steps, mean_regret_random - np.sqrt(var_regret_random), mean_regret_random + np.sqrt(var_regret_random), alpha=0.2, color='purple')

    if not np.isnan(mean_regret_causal_transDe[0]):
        plt.plot(steps, mean_regret_causal_transDe, label='Causal TransDe', color='blue')
        plt.fill_between(steps, mean_regret_causal_transDe - np.sqrt(var_regret_causal_transDe), mean_regret_causal_transDe + np.sqrt(var_regret_causal_transDe), alpha=0.2, color='blue')

    plt.xlabel('Steps')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid()
    #plt.title('source=('+dist_z_source[0] + ','+ dist_z_source[1] + ')\ntarget=('+ dist_z_target[0] + ', ' +dist_z_target[1]+')')
    plt.show()

def plot_z_comparasion(z, labels, name, z_index):
    x = np.arange(-4, 4, 0.1)
    p = norm.pdf(x, loc=0, scale=1)
    colors = ['#ee6c4d','#3d5a90' ]
    for i in range(len(z)):
        plt.hist(z[i], label=labels[i], alpha=0.6, color=colors[i], bins=50, density=True)
    plt.plot(x, p, label='prior', color='black', linestyle='dashed', alpha=0.6)
    plt.title("Distirbution of Z")
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.grid(False)
    
    #fig = plt.figure()
    #tikzplotlib_fix_ncols(fig)
    plt.show()
    #tikzplotlib.save("results/"+name)


def plot_total_regret_vs_freq(data, vae_freq, color_list):

    lables_list = [str(key) for key in data.keys()]

    for i in range(len(lables_list)):
        plt.plot(vae_freq, data[lables_list[i]], label=lables_list[i], color=color_list[i])
    # Explicitly set the x-axis limits to reverse order
    plt.xlim(min(vae_freq), max(vae_freq))
    plt.xlabel('num of VAE updating')
    plt.ylabel('Total regret')
    plt.legend()
    plt.grid()
    plt.show()


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def save_csv(vae_freq, total_regret_vs_vae_freq, name): 
    # store the results in csv file
    # data stricture:   
    # tranDe,randomVAE,sourceVAE,random, gradient step
    # x1,x2,x3,x4,x5
    # x1,x2,x3,x4,x5
    # x1,x2,x3,x4,x6
    num_update = [int(1000/freq) for freq in vae_freq]
    

    # Open a CSV file for writing

    keys = list(total_regret_vs_vae_freq.keys())
    keys.append('gradient_steps')

    # Open a CSV file for writing
    with open("results/"+name, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header (keys of the dictionary)
        writer.writerow(keys)

        # Write the rows (zip the ndarray values to create rows)
        rows = zip(*total_regret_vs_vae_freq.values(), num_update)
        for row in rows:
            writer.writerow(row)