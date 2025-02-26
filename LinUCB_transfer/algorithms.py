from typing import Any
import numpy as np
import copy
import random
from numpy.random import binomial
from numpy.random import normal
import torch
import torch.nn as nn
import torch.optim as optim





class LinUCB:
    """
    implementation for LinUCB agent
    """
    def __init__(self, k, num_features, alpha, leran_flag=True):
        self.k = k
        self.num_features = num_features
        self.alpha = alpha
        self.leran_flag = leran_flag

        self.A = [np.identity(num_features) for _ in range(k)]
        self.b = [np.zeros((num_features, 1)) for _ in range(k)]

        self.TA = [np.identity(num_features) for _ in range(k)]
        self.Tb = [np.zeros((num_features, 1)) for _ in range(k)]

    def act(self, context):
        p = [0] * self.k
        mu = [0] * self.k
        std = [0] * self.k
        theta = np.array([np.linalg.inv(self.A[i]).dot(self.b[i]) for i in range(self.k)])
        

        for a in range(self.k):
            x = np.array([context]).reshape(-1, 1)
            mu[a] = theta[a].T.dot(x)
            std[a] = self.alpha * np.sqrt(x.T.dot(np.linalg.inv(self.A[a])).dot(x))
            p[a] = mu[a] + std[a]

        chosen_arm = np.argmax(p)
        return chosen_arm

    def learn(self,action, reward, context):
        if self.leran_flag == False:
            return
    
        x = np.array([context]).reshape(-1, 1)
        self.A[action] = self.A[action] + x.dot(x.T)
        self.b[action] = self.b[action] + reward * x

    def train_act(self, context):
        p = [0] * self.k
        theta = np.array([np.linalg.inv(self.TA[i]).dot(self.Tb[i]) for i in range(self.k)])
        

        for a in range(self.k):
            x = np.array([context]).reshape(-1, 1)
            p[a] = theta[a].T.dot(x) + self.alpha * np.sqrt(x.T.dot(np.linalg.inv(self.TA[a])).dot(x))

        chosen_arm = np.argmax(p)
        return chosen_arm

    def train_learn(self,action, reward, context):
        x = np.array([context]).reshape(-1, 1)
        self.TA[action] = self.TA[action] + x.dot(x.T)
        self.Tb[action] = self.Tb[action] + reward * x
        a = 0
        
    def reset(self):
        self.transfer()

    def transfer(self):
        self.A = copy.copy(self.TA)
        self.b = copy.copy(self.Tb)



    

class CausalAgent():
    """
    A implementation of Causal method, where an encoder net is used to 
    approximate the posterior dist p(z|w) from observations of proxy variable
    """
    def __init__(self, k = 2, dim_proxy = 5, batch_size = 32, network = 'single_encoder', lr = '0.001', memory_size = 1000, beta = 1, alpha = 1, frequency= 1, device = 'cpu'):
        self.k = k
        self.dim_proxy = dim_proxy
        self.lr = lr
        self.memory_size = memory_size
        self.frequency = frequency
        self.network = network
        self.device = device
    
        if network == 'VAE':
            self.model = VAE(self.dim_proxy).to(self.device) 
        if network == 'single_encoder':
            self.model = Encoder(self.dim_proxy).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.memory = ReplayBuffer(self.memory_size)

        self.learned_a = []
        self.learned_b = []

        self.batch_size = batch_size
        self.beta = beta         # beta parameter for the KL divergence (set to 1 for standard VAE, 0 for reconstruction loss only)

        self.alpha = alpha       # alpha parameter for the reward function




    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        self.model.train()
        self.optimizer.zero_grad()
        # sample a batch of data from the memory
        batch = self.memory.sample(self.batch_size)
        batch = torch.tensor(batch).float().to(self.device)
        batch = torch.stack(list(batch), dim=0)        

        mu, log_var = self.model(batch)

        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu +  std*eps

        # compute the Kl divergence between the disribution of z and the standard normal distribution N(0,1)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # computing reconstruction loss by the true proxy generating mechanism p(w|z)
        log_px_z = -0.5 * ((batch - self.learned_a*z - self.learned_b)**2).mean(dim=0)

        loss = -(self.alpha * torch.mean(log_px_z)) + self.beta * kl_div
        
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def act(self, observation):
        
        self.model.eval()
        q = [0] * self.k

        # map high-dimensional observation to low-dimensional latent space
        obs = torch.tensor(observation, dtype=torch.float32).to(self.device)
        mu, log_var = self.model(obs)
    

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu +  std*eps
        
        # compute the q value for each action given the latent representation and transdered reward function from source domain
        for a in range(self.k):
           q[a] = self.RewardFunction(a, z)
        
        action = np.argmax(q)
        return action
    
    # reward function from source domain
    def RewardFunction(self, action, z):
        if (z >= 5 and  action== 0) or (z < 5 and action == 1):
            y = 1
        else:
            y = 0 
        return y

    def decoder(self, z):
        z = z.detach().numpy()
        x_hat = self.learned_a * z + self.learned_b #+ normal(size=5)
        x_hat = torch.tensor(x_hat, dtype=torch.float32)
        return x_hat

    def transfer(self, px_z):
        self.learned_a = torch.tensor(px_z[0], dtype=torch.float32).to(self.device)
        self.learned_b = torch.tensor(px_z[1], dtype=torch.float32).to(self.device)
    
    def reset(self):

        print('Reset the agent...')
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.memory = ReplayBuffer(self.memory_size)


class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    # store the experience into the memory
    def push(self, obs):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        

        self.memory[self.position] = obs
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size x' from the replay memory and returns an array
        """
        sample = random.sample(tuple(self.memory), batch_size)
        return sample



class Encoder(nn.Module):
    def __init__(self, x_dims):
        super(Encoder, self).__init__()
        self.fc_mu = nn.Linear(x_dims, 1)    # For mean
        self.fc_log_var = nn.Linear(x_dims, 1)   # For log variance

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)  # Encoder predicts log variance     

        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, x_dims):
      super(Decoder, self).__init__()
      self.fc_re = nn.Linear(1, x_dims)

    def forward(self, x):
      x_restore = self.fc_re(x)
      return x_restore
    
class VAE(nn.Module):
    def __init__(self, x_dims):
      super(VAE, self).__init__()
      self.encoder = Encoder(x_dims)
      self.decoder = Decoder(x_dims)


    def forward(self, x):
      mu, log_var = self.encoder(x)
      return mu, log_var

