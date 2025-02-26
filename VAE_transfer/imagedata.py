import numpy as np
import random
import torch
import pandas as pd
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
class ImageDataset(Dataset):
    def __init__(self, images, x, t, y,z):
        self.length = x.shape[0]
        x_dim = x.shape[1]
        self.images = images
        self.t = t
        self.X = x
        self.y = y
        self.z = z

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'X': self.X[idx],
            't': self.t[idx],
            'y': self.y[idx],
            'z': self.z[idx]
        }
    def __len__(self):
        return self.length
    
    def apply_mask(self, mask):
        """
        Applies a mask to the dataset and returns a new ImageDataset
        containing only the elements that are not masked out.

        Parameters:
        mask (list or numpy array): A boolean list or array indicating
                                    which elements to keep (True) or discard (False).

        Returns:
        ImageDataset: A new dataset with only the unmasked elements.
        """
        mask = torch.tensor(mask, dtype=torch.bool)  # Ensure mask is a boolean tensor

        # Filter the data using the mask
        masked_images = self.images[mask]
        masked_X = self.X[mask]
        masked_t = self.t[mask]
        masked_y = self.y[mask]
        masked_z = self.z[mask]

        # Create and return a new ImageDataset with the filtered data
        return ImageDataset(masked_images, masked_X, masked_t, masked_y, masked_z)

    def dataset_to_dataframe(self):

        df = pd.DataFrame(torch.cat([self.z,self.X,self.t,self.y],1).detach().numpy().squeeze(),columns=["z"+str(i) for i in range(self.z.shape[1])] + ["x0","t","y"])
    
        return df
    
    def sample(self, size):
        random_mask = [1]*size + [0]*(self.length - size)
        random.shuffle(random_mask)

        return self.apply_mask(mask=random_mask)

    def shuffle(self):
        indices = torch.randperm(self.length)
        self.images = self.images[indices]
        self.X = self.X[indices]
        self.t = self.t[indices]
        self.y = self.y[indices]
        self.z = self.z[indices]        

    def sort_by_z(self):
        # Get sorted indices
        sorted_indices = torch.argsort(self.z[:,0])

        # Reorder the dataset using the sorted indices
        sorted_image= self.images[sorted_indices]
        sorted_X= self.X[sorted_indices]
        sorted_t= self.t[sorted_indices]
        sorted_y= self.y[sorted_indices]
        sorted_z= self.z[sorted_indices]

        # Create and return a new sorted dataset
        return ImageDataset(sorted_image, sorted_X, sorted_t, sorted_y, sorted_z)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def generate_image_data(num_samples, zdim, generator, t_a, t_b, y_a0, y_b0, y_a1, y_b1, c_x, s_x):
    z = torch.randn(num_samples, zdim)
    with torch.no_grad():
        image_expectations = (1+generator(z[:,:,None,None]))/2
        images = dist.Bernoulli(image_expectations).sample()#This way binary data
        #image_means = generator(z[:,:,None,None])
        #images = image_means# + torch.randn_like(image_means)*0.05 # <- this way continuous data
    z_temp = z[:,0][:,None].detach().numpy()#Use the first dimension for prediction of ordinary variables
    x = torch.Tensor(np.random.normal(np.tile(c_x, (num_samples,1))*z_temp,
                         np.tile(s_x, (num_samples,1)),(num_samples, 1)))
    t = (np.random.random((num_samples, 1)) < sigmoid(t_a*z_temp + t_b)).astype(int)
    #y = torch.Tensor((np.random.random((num_samples, 1)) < sigmoid(y_a1*z_temp + y_b1)).astype(int)*t \
    #    + (np.random.random((num_samples, 1)) < sigmoid(y_a0*z_temp + y_b0)).astype(int)*(1-t))
    y = torch.Tensor(z_temp*t + np.random.randn(num_samples, 1))
    t = torch.Tensor(t)
    dataset = ImageDataset(images, x, t, y, z)
    return z, images, x, t, y, dataset