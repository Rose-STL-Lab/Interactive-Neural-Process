import torch
import torch.nn as nn 
from scipy.stats import multivariate_normal
import numpy as np


def mae_loss(y_pred, y_true):
    loss = torch.abs(y_pred - y_true)
    loss[loss != loss] = 0
    return loss.mean()

def mae_metric(y_pred, y_true):
    loss = np.abs(y_pred - y_true)
    loss[loss != loss] = 0
    loss = np.mean(loss)
    return loss

def rmse_metric(y_pred, y_true):
    loss0 = (y_pred-y_true)**2
    loss0[loss0 != loss0] = 0
    loss = np.sqrt(np.mean(loss0))
    return loss

def kld_gaussian_loss(z_mean_all, z_var_temp_all, z_mean_context, z_var_temp_context): 
    """Analytical KLD between 2 Gaussians."""
    mean_q, var_q, mean_p, var_p = z_mean_all, 0.1+ 0.9*torch.sigmoid(z_var_temp_all), z_mean_context,  0.1+ 0.9*torch.sigmoid(z_var_temp_context)
    std_q = torch.sqrt(var_q)
    std_p = torch.sqrt(var_p)
    p = torch.distributions.Normal(mean_p, std_p)
    q = torch.distributions.Normal(mean_q, std_q)
    return torch.distributions.kl_divergence(p, q).sum()

def maxentropy(pred):
        pred = torch.exp(pred) - 1.
        pred = pred.detach().cpu().numpy()
        pred = np.transpose(pred, (1, 0, 2))
        pred = pred.reshape(pred.shape[0],-1)

        # mean_std = torch.mean(torch.std(pred, 1))
        mean = np.mean(pred,0)
        cov = np.cov(pred.T)
        score = multivariate_normal(mean, cov,allow_singular=True).entropy()

        return score
