from torch.utils.data import Dataset
import torch
import pickle
import os
import re

class CEVAEDataset(Dataset):
    def __init__(self, df):
        x_cols = [c for c in df.columns if c.startswith("x")]
        self.X = torch.Tensor(df[x_cols].to_numpy())
        self.t = torch.Tensor(df["t"].to_numpy()[:,None])
        self.y = torch.Tensor(df["y"].to_numpy()[:,None])
        self.length = len(df)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            't': self.t[idx],
            'y': self.y[idx]
        }
    def __len__(self):
        return self.length


def load_dfs(main_folder, sub_folder, param_times=None):
    dfs = {}
    datasets = {}
    for file in os.listdir("data/{}/{}/".format(main_folder, sub_folder)):
        match = re.search(r"df_([^_]*)_(\d*)", file)
        if match:
            if not match.group(1) in dfs:
                dfs[match.group(1)] = {}
                datasets[match.group(1)] = {}
            with open("data/{}/{}/{}".format(main_folder,sub_folder,file), "rb") as file:
                dfs[match.group(1)][int(match.group(2))] = pickle.load(file)
                datasets[match.group(1)][int(match.group(2))] = CEVAEDataset(dfs[match.group(1)][int(match.group(2))])
        else:
            match = re.search(r"df_([^_]*)", file)
            with open("data/{}/{}/{}".format(main_folder,sub_folder,file), "rb") as file:
                dfs[match.group(1)] = {}
                datasets[match.group(1)] = {}
                df =  pickle.load(file)
                for i in range(param_times):
                    dfs[match.group(1)][i] = df
                    datasets[match.group(1)][i] = CEVAEDataset(df)
    return dfs, datasets