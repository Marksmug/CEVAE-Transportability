import numpy as np
import torch 
import random
from pandas import DataFrame
from torch.optim import Adam
import torch.distributions as dist
        

def eva_random(data, steps):
    #data = data.sample(steps)
    if isinstance(data, DataFrame):
        z_data = np.array(data.loc[:, 'z'])
        
    else:
        z_data = np.array(data.z[:,0])
    reward = [0]
    regret = [0]

    for z in z_data:
        
        
        t = random.randint(0,1)
        eps = np.random.normal()
        #y = t + z - t*z + eps
        y = t*z + eps

        #optimal = max(1 + eps, z + eps)
        optimal = max(eps, z + eps)
        r = optimal - y
        current_regret = regret[-1] + r
        current_reward =  reward[-1] + y
        reward.append(current_reward)
        regret.append(current_regret)

    return reward, regret


def eva_negative(data, model, steps):
    #data = data.sample(steps)
    x_data = np.array(data.iloc[:, 1:25])
    z_data = np.array(data.loc[:, 'z'])
    regret = [0]
    reward = [0]
    num_opt_source = 0

    for x, z in zip(x_data, z_data):

        z = float(z)
        x = np.array([x]).reshape(1, -1)
        t0 = np.array([0]).reshape(-1, 1)
        t1 = np.array([1]).reshape(-1, 1)
        xt0 = np.concatenate([x, t0], 1)
        xt1 = np.concatenate([x, t1], 1)
        torch.Tensor(x)
        # estimate y given t=0 and t=1
        y0 = model(torch.Tensor(xt0))
        y1 = model(torch.Tensor(xt1))

        if y0 > y1:
            t = 0
        else:
            t = 1
            num_opt_source += 1
    
        # give the reward y from the generating process
        eps = np.random.normal()
        #y = t + z - t*z + eps
        y = t*z + eps

        current_reward =  reward[-1] + y
        reward.append(current_reward)
        #optimal = max(1 + eps, z + eps)
        optimal = max(eps, z + eps)
        r = optimal - y
        current_regret = regret[-1] + r
        regret.append(current_regret)
 
    return reward, regret



def eva_causal(data, vae, beta = 1, vae_freq=10, lr=0.01, batch_size=32, train=True, update_decoder = True):
    #data = data.sample(steps)
    x_dim = vae.x_dim
    x_data = np.array(data.iloc[:, 1:x_dim+1])
    z_data = np.array(data.loc[:, 'z'])
    reward = [0]
    regret = [0]
    vae_z = []
    true_z = []

    memory = ReplayBuffer(capacity=1000)

    if update_decoder: 
        optimizer = Adam(list(vae.encoder.parameters())+ list(vae.decoder.parameters()), lr=lr)
    else:
        optimizer = Adam(list(vae.encoder.parameters()), lr=lr)
    i = 0
    for x, z in zip(x_data, z_data):
        i += 1

        # Interative step
        x = np.array([x]).reshape(1, -1)
        z_pred, z_std = vae.encoder(torch.Tensor(x))
        latent_z = vae.reparameterize(z_pred, z_std)
        #latent_z = z_pred.detach().numpy()
        

        t0 = torch.tensor([0]).view(1,1)
        t1 = torch.tensor([1]).view(1,1)

        # estimate y given t=0 and t=1
        y0_mu, y0_std = vae.decoder.y_nn(latent_z, t0)
        y0 = dist.Normal(loc=y0_mu, scale=y0_std).sample()
        y1_mu, y1_std = vae.decoder.y_nn(latent_z, t1)
        y1 = dist.Normal(loc=y1_mu, scale=y1_std).sample()


        if y0 > y1:
            t = 0
        else:
            t = 1

        # recieving the reward y from the generating process
        z = float(z)
        eps = np.random.normal()
        #y = t + z - t*z + eps
        #optimal = max(1 + eps, z + eps)
        y = t*z + eps
        optimal = max(eps, z + eps)
        
        
        r = optimal - y
        current_regret = regret[-1] + r
        current_reward =  reward[-1] + y
        reward.append(current_reward)
        regret.append(current_regret)
        vae_z.append(float(latent_z[:,0]))
        true_z.append(z)

        # Learning step
        memory.push([x, t, y])
        if i % vae_freq == 0 and train:
            loss = vae_learning(vae, memory, optimizer, batch_size, beta)
            #print('The current loss is ', loss)

    return reward, regret, vae_z, true_z

def eva_causal_image(device, data, vae, beta = 1, vae_freq=10, lr=0.01, batch_size=32, train=True, update_decoder = True):
    #data = data.sample(steps)
    x_dim = vae.x_dim
    image_data = np.array(data.images)
    x_data = np.array(data.X)
    z_data = np.array(data.z)
    reward = [0]
    regret = [0]
    vae_z = []
    true_z = []

    memory = ReplayBuffer(capacity=1000)

    if update_decoder: 
        optimizer = Adam(list(vae.encoder.parameters())+ list(vae.decoder.parameters()), lr=lr)
    else:
        optimizer = Adam(list(vae.encoder.parameters()), lr=lr)
    i = 0
    for image, x, z in zip(image_data, x_data, z_data):
        i += 1
        
        # Interative step
        image = np.expand_dims(image,axis=0)
        x = np.array([x]).reshape(1, -1)
        z_pred, z_std = vae.encoder(torch.Tensor(image).to(device),torch.Tensor(x).to(device))
        latent_z = vae.reparameterize(z_pred, z_std)
        #latent_z = z_pred.detach().numpy()
        
        #not necessary for CEVAE structure
        t0 = torch.tensor([0]).view(1,1).to(device)
        t1 = torch.tensor([1]).view(1,1).to(device)

        # estimate y given t=0 and t=1
        

        y0 = vae.decoder.estimate_causal_effect(latent_z, t0)
        y1 = vae.decoder.estimate_causal_effect(latent_z, t1)


        if y0 > y1:
            t = 0
        else:
            t = 1

        # recieving the reward y from the generating process
        eps = np.random.normal()
        #y = t + z - t*z + eps
        #optimal = max(1 + eps, z + eps)
        y = t*z[0] + eps
        optimal = max(eps, z[0] + eps)
        
        
        r = optimal - y
        current_regret = regret[-1] + r
        current_reward =  reward[-1] + y
        reward.append(current_reward)
        regret.append(current_regret)

        vae_z.append(float(latent_z[:,0].detach().cpu().numpy()))
        #z = z[0]í
        #vae_z.append(latent_z)
        true_z.append(z[0])

        # Learning step
        memory.push([image, x, t, y])
        if i % vae_freq == 0 and train:
            loss = vae_learning_image(device, vae, memory, optimizer, batch_size, beta)
            #print('The current loss is ', loss)

    return reward, regret, vae_z, true_z


def vae_learning_image(device, vae, memory, optimizer, batch_size=32, beta=1):
    if len(memory) < batch_size:
        return   
    obs = memory.sample(batch_size)
    obs = list(zip(*obs))
    image_shape = np.array(obs[0][0]).shape
    image = torch.Tensor(obs[0]).view(batch_size, image_shape[1], image_shape[2], image_shape[3]).to(device)
    x = torch.Tensor(obs[1]).view(batch_size, vae.x_dim).to(device)
    t = torch.Tensor(obs[2]).view(batch_size, 1).to(device)
    y = torch.Tensor(obs[3]).view(batch_size, 1).to(device)
    image_mean, image_std, z_mean, z_std, x_pred, x_std, t_pred, y_pred, y_std = vae(image, x, t)
    kld = kld_loss(z_mean, z_std)
    #image_loss = -dist.Normal(loc=image_mean, scale = image_std).log_prob(image).sum()
    image_loss = -dist.Bernoulli(logits=image_mean).log_prob(image).mean(0).sum()
    x_loss = -dist.Normal(loc=x_pred, scale = x_std).log_prob(x).mean(0).sum()
    if vae.decoder.p_t_z_nn:
        t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).mean(0).sum()
    else:
        t_loss = torch.tensor(0).to(device)
    y_loss = -dist.Normal(loc=y_pred, scale = y_std).log_prob(y).mean(0).sum() 
    loss = beta*kld + image_loss + x_loss + y_loss + t_loss
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients
    torch.nn.utils.clip_grad_value_(vae.parameters(), clip_value=1)

    optimizer.step()
    return loss.item()





def vae_learning(vae, memory, optimizer, batch_size=32, beta=1):
    if len(memory) < batch_size:
        return   
    obs = memory.sample(batch_size)
    obs = list(zip(*obs))
    x = torch.Tensor(obs[0]).view(batch_size, vae.x_dim)
    t = torch.Tensor(obs[1]).view(batch_size, 1)
    y = torch.Tensor(obs[2]).view(batch_size, 1)
    z_mean, z_std, x_pred, x_std, t_pred, y_mu, y_std = vae(x, t)
    kld, x_loss, y_loss, t_loss = get_losses(z_mean, z_std, x_pred, x_std, x, vae.x_mode, t_pred, t, y_mu, y_std, y)
    loss = beta*kld + x_loss + y_loss + t_loss
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients
    torch.nn.utils.clip_grad_value_(vae.parameters(), clip_value=1)

    optimizer.step()
    return loss.item()

def kld_loss(mu, std):
    #Note that the sum is over the dimensions of z as well as over the units in the batch here
    var = std.pow(2)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, 1).sum()
    return kld

def y_log_prob_loss(y, y_mu, y_std):
    y_loss = -dist.Normal(loc=y_mu, scale=y_std).log_prob(y).sum()
    return y_loss

def get_losses(z_mean, z_std, x_pred, x_std, x, x_mode,t_pred, t, y_mu, y_std, y):
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
        t_loss = torch.tensor(0)
    y_loss = y_log_prob_loss(y, y_mu, y_std)
    return kld, x_loss, y_loss, t_loss

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
        Samples batch_size x' from the replay memory and returns a list
        """
        sample = random.sample(tuple(self.memory), batch_size)
        return sample