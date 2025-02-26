import torch
from torch import nn, optim
import torch.distributions as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm 


class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        layers.append(nn.Linear(sizes[0],sizes[1]))
        for in_size, out_size in zip(sizes[1:], sizes[2:]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_size, out_size))
        if final_activation is not None:
            layers.append(final_activation)
        self.length = len(layers)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)
        
    def __len__(self):
        return self.length

class y_nn(nn.Module):
    def __init__(
            self,
            z_dim,
            p_y_zt_nn_layers,
            p_y_zt_nn_width,
            y_separate_heads,
            y_loss_type
    ):
        super().__init__()
        self.z_dim = z_dim
        self.p_y_zt_nn_layers = p_y_zt_nn_layers
        self.p_y_zt_nn_width = p_y_zt_nn_width
        self.y_separate_heads = y_separate_heads
        self.y_loss_type = y_loss_type

        if y_separate_heads:
            self.y0_nn = FullyConnected([z_dim] + p_y_zt_nn_layers * [p_y_zt_nn_width] + [y_loss_type])
            self.y1_nn = FullyConnected([z_dim] + p_y_zt_nn_layers * [p_y_zt_nn_width] + [y_loss_type])
        else:
            self.y_nn = FullyConnected([z_dim+1] + p_y_zt_nn_layers * [p_y_zt_nn_width] + [y_loss_type])

    def forward(self, z, t):
        if self.y_separate_heads:
            if self.y_loss_type == 2:
                y0_dist = self.y0_nn(z)
                y0_mu = y0_dist[:, :1]
                y0_std = torch.exp(y0_dist[:, 1:])   
                y1_dist = self.y1_nn(z)
                y1_mu = y1_dist[:, :1]
                y1_std = torch.exp(y1_dist[:, 1:])     
                y_std = y1_std*t + y0_std*(1-t)    
            else:
                y0_mu = self.y0_nn(z)
                y1_mu = self.y1_nn(z)
                y_std = 0
            y_mu = y1_mu*t + y0_mu*(1-t) 
        else:
            if self.y_loss_type == 2:
                y_dist = self.y_nn(torch.cat([z,t], 1))
                y_mu = y_dist[:, :1]
                y_std = torch.exp(y_dist[:, 1:])
            else:
                y_mu = self.y_nn(torch.cat([z,t], 1))
                y_std = 0

        return y_mu, y_std

class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        device,
        p_t_z_nn,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        p_t_z_nn_layers,
        p_t_z_nn_width,
        p_y_zt_nn_layers,
        p_y_zt_nn_width,
        x_mode,
        y_loss_type,
        y_separate_heads
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.p_t_z_nn = p_t_z_nn
        self.p_x_z_nn_layers = p_x_z_nn_layers
        self.p_x_z_nn_width = p_x_z_nn_width
        self.p_t_z_nn_layers = p_t_z_nn_layers
        self.p_t_z_nn_width = p_t_z_nn_width
        self.p_y_zt_nn_layers = p_y_zt_nn_layers
        self.p_y_zt_nn_width = p_y_zt_nn_width
        self.y_separate_heads = y_separate_heads
        
        #Can be used as a linear predictor if num_hidden=0
        self.n_x_estimands = sum([1 if m==0 or m==2 else m for m in x_mode])
        #for each x we have the possible std estimator also for simplicity, possibly not used
        self.x_nn = FullyConnected([z_dim] + p_x_z_nn_layers*[p_x_z_nn_width] + [(self.n_x_estimands)*2])

        if self.p_t_z_nn:
            self.t_nn = FullyConnected([z_dim] + p_t_z_nn_layers*[p_t_z_nn_width] + [1])
        self.y_nn = y_nn(z_dim, p_y_zt_nn_layers, p_y_zt_nn_width, y_separate_heads, y_loss_type)
        
        self.to(device)
        
    def forward(self,z, t):
        x_res = self.x_nn(z)
        y_mu, y_std = self.y_nn(z, t)
        if self.p_t_z_nn:
            t_pred = self.t_nn(z)
        else:
            t_pred = None
        x_pred = x_res[:,:self.n_x_estimands]
        x_std = torch.exp(x_res[:,self.n_x_estimands:])
        return x_pred,x_std, t_pred, y_mu, y_std, 
    
    

class Encoder(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        q_z_nn_layers,
        q_z_nn_width
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.q_z_nn_layers = q_z_nn_layers
        self.q_z_nn_width = q_z_nn_width
        
        # q(z|x,t,y)
        self.q_z_nn = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        self.to(device)
        
    def forward(self,x):
        z_res = self.q_z_nn(x)
        z_pred = z_res[:,:self.z_dim]
        z_std = torch.exp(z_res[:,self.z_dim:])
        return z_pred, z_std
    
class ProxyVAE(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        p_t_z_nn,
        p_t_z_nn_layers,
        p_t_z_nn_width,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        p_y_zt_nn_layers,
        p_y_zt_nn_width,
        q_z_nn_layers,
        q_z_nn_width,
        x_mode, #a list, 0 for continuous (Gaussian), 2 or more for categorical distributions (usually 2 or 0)
        y_loss_type,
        y_separate_heads
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.x_mode = x_mode
        self.y_separate_heads = y_separate_heads
        assert all([x_m == 0 or x_m > 1 for x_m in x_mode])
        assert len(x_mode) == x_dim
        
        self.encoder = Encoder(
            x_dim,
            z_dim,
            device,
            q_z_nn_layers,
            q_z_nn_width
        )
        
        self.decoder = Decoder(
            x_dim,
            z_dim,
            device,
            p_t_z_nn,
            p_t_z_nn_layers,
            p_t_z_nn_width,
            p_x_z_nn_layers,
            p_x_z_nn_width,
            p_y_zt_nn_layers,
            p_y_zt_nn_width,
            x_mode,
            y_loss_type,
            y_separate_heads
        )
        
        self.to(device)
        self.float()
        
    def reparameterize(self, mean, std):
        # samples from unit norm and does reparam trick
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def forward(self, x, t):
        z_mu, z_std = self.encoder(x)
        #TODO: works at least for z_dim=1, maybe errors if z_dim>1
        z = self.reparameterize(z_mu, z_std)
        x_pred, x_std, t_pred, y_mu, y_std = self.decoder(z, t)
        
        return z_mu, z_std, x_pred, x_std, t_pred, y_mu, y_std
    
    def sample(self,n):
        # only sample for x
        different_modes = list(set(self.x_mode))
        x_same_mode_indices = dict()
        for mode in different_modes:
            x_same_mode_indices[mode] = [i for i,m in enumerate(self.x_mode) if m==mode]

        z_sample = torch.randn(n, self.z_dim).to(self.device)
        x_pred, x_std = self.decoder(z_sample)
        x_sample = np.zeros((n, self.x_dim))

        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(self.x_mode):
            if mode==0:
                x_sample[:,i] = dist.Normal(loc=x_pred[:,pred_i], scale=x_std[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            elif mode==2:
                x_sample[:,i] = dist.Bernoulli(logits=x_pred[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            else:
                x_sample[:,i] = dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).sample().detach().numpy()
                pred_i += mode
        
        return z_sample, x_sample
    
    def recons(self, x):
        different_modes = list(set(self.x_mode))
        x_same_mode_indices = dict()
        for mode in different_modes:
            x_same_mode_indices[mode] = [i for i,m in enumerate(self.x_mode) if m==mode]

        _, _, x_pred, x_std = self.forward(torch.Tensor(x))
        x_sample = np.zeros((len(x), self.x_dim))

        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(self.x_mode):
            if mode==0:
                x_sample[:,i] = dist.Normal(loc=x_pred[:,pred_i], scale=x_std[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            elif mode==2:
                x_sample[:,i] = dist.Bernoulli(logits=x_pred[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            else:
                x_sample[:,i] = dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).sample().detach().numpy()
                pred_i += mode
        
        return x_sample

def train_decoder(device, plot_curves, print_logs,
              train_loader, num_epochs, lr_start, lr_end, x_dim, z_dim,p_t_z_nn=False,p_t_z_nn_layers=3,
              p_t_z_nn_width=10, p_x_z_nn_layers=3, p_x_z_nn_width=10, p_y_nn_layers=3, p_y_nn_width=10,
              q_z_nn_layers=3, q_z_nn_width=10, x_mode=[0], y_loss_type=2, y_separate_heads=False):
    
    model = ProxyVAE(x_dim, z_dim, device, p_t_z_nn, p_t_z_nn_layers, p_t_z_nn_width, p_x_z_nn_layers, p_x_z_nn_width, p_y_nn_layers, p_y_nn_width, q_z_nn_layers, q_z_nn_width, x_mode, y_loss_type, y_separate_heads)
    optimizer = Adam(model.decoder.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    losses = {"total": [], "x": [], "t":[], "y":[]}
    
    different_modes = list(set(x_mode))
    x_same_mode_indices = dict()
    for mode in different_modes:
        x_same_mode_indices[mode] = [i for i,m in enumerate(x_mode) if m==mode]
    
    def get_losses(x_pred, x_std, x, t_pred, t, y_mu, y_std, y):
        x_loss = 0
        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        #for i,mode in enumerate(x_mode):
        #    if mode==0:
        #        x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).mean(0).sum()
        #        pred_i += 1
        #    elif mode==2:
        #        x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).mean(0).sum()
        #        pred_i += 1
        #    else:
        #        x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).mean(0).sum()
        #        pred_i += mode
        for i,mode in enumerate(x_mode):
            if mode==0:
                x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).sum()
                pred_i += 1
            elif mode==2:
                x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).sum()
                pred_i += 1
            else:
                x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).sum()
                pred_i += mode
            if t_pred is not None:
                t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).mean(0).sum()
            else:
                t_loss = torch.tensor(0).to(device)
        y_loss = y_log_prob_loss(y, y_mu, y_std)
        return x_loss, y_loss, t_loss
    
    for epoch in range(num_epochs):
        i = 0
        epoch_loss = 0
        epoch_x_loss = 0
        epoch_t_loss = 0
        epoch_y_loss = 0
        if print_logs:
            print("Epoch {}:".format(epoch))
        for data in train_loader:
            data = data.to(device)
            z = data[:, :z_dim]
            x = data[:, z_dim:x_dim+z_dim]
            t = data[:, x_dim+z_dim].unsqueeze(1)
            y = data[:, x_dim+z_dim+1].unsqueeze(1)
            x_pred, x_std, t_pred, y_mu, y_std = model.decoder(z, t)
            x_loss, y_loss, t_loss = get_losses(x_pred, x_std, x, t_pred, t, y_mu, y_std, y)
            loss = x_loss + y_loss + t_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 and print_logs:
                print("Sample batch loss: {}".format(loss))
            epoch_loss += loss.item()
            epoch_x_loss += x_loss.item()
            epoch_t_loss += t_loss.item()
            epoch_y_loss += y_loss.item()
        
        losses['total'].append(epoch_loss)
        losses['x'].append(epoch_x_loss)
        losses["t"].append(epoch_t_loss)
        losses['y'].append(epoch_y_loss)
        scheduler.step()
  
        if print_logs:
            #print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("Epoch loss: {}".format(epoch_loss))
            print("x: {}, t: {}, y: {}".format(epoch_x_loss, epoch_t_loss, epoch_y_loss))
            
            
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(losses['x'])
    ax[0,1].plot(losses['t'])
    ax[1,0].plot(losses['y'])
    ax[1,1].plot(losses['total'])
    ax[0,0].set_title("x loss")
    ax[0,1].set_title("t loss")
    ax[1,0].set_title("y loss")
    ax[1,1].set_title("total loss")
    plt.show()
    
    return model.decoder, losses

def train_ProxyVAE(device, plot_curves, print_logs,
              train_loader, num_epochs, lr_start, lr_end, x_dim, z_dim,p_t_z_nn=False,p_t_z_nn_layers=3,
              p_t_z_nn_width=10, p_x_z_nn_layers=3, p_x_z_nn_width=10, p_y_nn_layers=3, p_y_nn_width=10,
              q_z_nn_layers=3, q_z_nn_width=10, x_mode=[0], y_loss_type=2, y_separate_heads=False):
    
    model = ProxyVAE(x_dim, z_dim, device, p_t_z_nn, p_t_z_nn_layers, p_t_z_nn_width, p_x_z_nn_layers, p_x_z_nn_width, p_y_nn_layers, p_y_nn_width, q_z_nn_layers, q_z_nn_width, x_mode, y_loss_type, y_separate_heads)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    losses = {"total": [], "kld": [], "x": [], "t":[], "y":[]}
    
    def kld_loss(mu, std):
        #Note that the sum is over the dimensions of z as well as over the units in the batch here
        var = std.pow(2)
        kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
        return kld
    
    different_modes = list(set(x_mode))
    x_same_mode_indices = dict()
    for mode in different_modes:
        x_same_mode_indices[mode] = [i for i,m in enumerate(x_mode) if m==mode]
    
    def get_losses(z_mean, z_std, x_pred, x_std, x, t_pred, t, y_mu, y_std, y):
        kld = kld_loss(z_mean,z_std)
        x_loss = 0
        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(x_mode):
            if mode==0:
                x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).sum()
                pred_i += 1
            elif mode==2:
                x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).sum()
                pred_i += 1
            else:
                x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).sum()
                pred_i += mode
        if t_pred is not None:
            t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
        else:
            t_loss = torch.tensor(0).to(device)
        y_loss = y_log_prob_loss(y, y_mu, y_std)
        return kld, x_loss, y_loss, t_loss
    
    for epoch in range(num_epochs):
        i = 0
        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_x_loss = 0
        epoch_t_loss = 0
        epoch_y_loss = 0
        if print_logs:
            print("Epoch {}:".format(epoch))
        for data in train_loader:
            data = data.to(device)
            x = data[:, :x_dim]
            t = data[:, x_dim].unsqueeze(1)
            y = data[:, x_dim+1].unsqueeze(1)
            z_mean, z_std, x_pred, x_std, t_pred, y_mu, y_std = model(x, t)
            kld, x_loss, y_loss, t_loss = get_losses(z_mean, z_std, x_pred, x_std, x, t_pred, t, y_mu, y_std, y)
            loss = kld + x_loss + y_loss + t_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 and print_logs:
                print("Sample batch loss: {}".format(loss))
            epoch_loss += loss.item()
            epoch_kld_loss += kld.item()
            epoch_x_loss += x_loss.item()
            epoch_t_loss += t_loss.item()
            epoch_y_loss += y_loss.item()
        
        losses['total'].append(epoch_loss)
        losses['kld'].append(epoch_kld_loss)
        losses['x'].append(epoch_x_loss)
        losses["t"].append(epoch_t_loss)
        losses['y'].append(epoch_y_loss)
        scheduler.step()
  
        if print_logs:
            #print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("Epoch loss: {}".format(epoch_loss))
            print("x: {}, t: {}, y: {}, kld: {}".format(epoch_x_loss, epoch_t_loss, epoch_y_loss, epoch_kld_loss))
            
            
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(losses['x'])
    ax[0,1].plot(losses['kld'])
    ax[1,0].plot(losses['y'])
    ax[1,1].plot(losses['total'])
    ax[0,0].set_title("x loss")
    ax[0,1].set_title("kld loss")
    ax[1,0].set_title("y loss")
    ax[1,1].set_title("total loss")
    plt.show()
    
    return model, losses

def trainZTtoYmodel(device, z, t, y, lr_start, lr_end, num_epochs ,layers=3, width=10, loss_type=1, y_separate_heads=False):
    #loss_type = 1 uses MSE loss and 2 uses log_prob loss 

    # cover z,t,y to array (len(), 1)
    z = np.array(z).reshape(-1,1)
    t = np.array(t).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    z_dim = z.shape[1]
    #model = FullyConnected([z_dims+1] + layers*[width] + [loss_type])
    model = y_nn(z_dim, layers, width, y_separate_heads, loss_type)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    
    dataset = GenericDataset(np.concatenate([z,t,y],1))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            z = data[:, 0].unsqueeze(1)
            t = data[:, 1].unsqueeze(1)
            y = data[:, 2].unsqueeze(1)    
            
            if loss_type == 1:
                y_pred,_ = model(z, t)
                loss = RMSE_loss(y, y_pred)
            else:
                y_mu, y_std= model(z, t)
                loss = y_log_prob_loss(y, y_mu, y_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Train loss in epoch {epoch} is {epoch_loss}")
        losses.append(epoch_loss)
    plt.plot(losses)
    plt.title("Epoch loss")
    return model

def trainZtoXmodel(device, z, x, x_dim, z_dim, x_mode,  lr_start, lr_end, num_epochs ,layers=3, width=10):
    #Returns a model that is 
    n_x_estimands = sum([1 if m==0 or m==2 else m for m in x_mode])

    model = FullyConnected([z_dim] + layers*[width] + [n_x_estimands*2])

    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    
    z = np.array(z).reshape(-1,1)
    x = np.array(x)

    dataset = GenericDataset(np.concatenate([z, x],1))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            z = data[:,[0]]
            x = data[:,1:]
            
            x_dist = model(z)
            x_mu = x_dist[:,:n_x_estimands]
            x_std = torch.exp(x_dist[:,n_x_estimands:])          
            loss = x_loss(x_mu, x_std, x, x_mode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Train loss in epoch {epoch} is {epoch_loss}")
        losses.append(epoch_loss)
    plt.plot(losses)
    plt.title("Epoch loss")
    return model



def trainXTtoYmodel(device, x, t, y, lr_start, lr_end, num_epochs ,layers=3, width=10, batch_size=32, loss_type = 1):
    #Returns a model of p(Y|X, T)
    dims_xt = x.shape[1]+1
    model = FullyConnected([dims_xt] + [width]*layers + [loss_type])
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    # cover x,t,y to array 
    x = np.array(x) # do not need reshape
    t = np.array(t).reshape(-1,1)
    y = np.array(y).reshape(-1,1)

    dataset = GenericDataset(np.concatenate([x,t,y],1))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            xt = data[:,0:dims_xt]
            y = data[:,dims_xt].unsqueeze(1)
            
            if loss_type == 1:
                y_pred = model(xt)
                loss = RMSE_loss(y, y_pred)
            else:
                y_dist= model(xt)
                y_mu = y_dist[:, :1]
                y_std = torch.exp(y_dist[:, 1:])
                loss = y_log_prob_loss(y, y_mu, y_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            
        print(f"Train loss in epoch {epoch} is {epoch_loss}")
        losses.append(epoch_loss)
    plt.plot(losses)
    plt.title("Epoch loss")
    return model



class GenericDataset(Dataset):
    def __init__(self, data):
        self.X = torch.Tensor(data)
        self.length = len(data)
    
    def __getitem__(self, idx):
        return self.X[idx]
        
    def __len__(self):
        return self.length
    
def RMSE_loss(y, ypred):
    loss = torch.sqrt((y - ypred)**2).sum()
    #loss = F.mse_loss(y, ypred, reduction="mean")
    #loss = torch.sqrt(mse)
    return loss

def y_log_prob_loss(y, y_mu, y_std):
    y_loss = -dist.Normal(loc=y_mu, scale=y_std).log_prob(y).sum()
    return y_loss

def x_loss(x_pred, x_std, x, x_mode):
    x_loss = 0
    pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
    for i,mode in enumerate(x_mode):
        if mode==0:
            x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).sum()
            pred_i += 1
        elif mode==2:
            x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).sum()
            pred_i += 1
        else:
            x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).sum()
            pred_i += mode
    return x_loss

"""
def x_loss(x_pred, x_std, x, x_mode):
    eps = torch.randn_like(x_std)
    x_hat = eps.mul(x_std).add(x_pred)

    loss = torch.sqrt((x-x_hat)**2).sum()

    return loss
"""