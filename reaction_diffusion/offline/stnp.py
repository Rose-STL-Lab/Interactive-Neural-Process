#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
from numpy.random import binomial
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import torch.nn as nn
from sklearn import preprocessing
from scipy.stats import multivariate_normal


# In[38]:


device = torch.device("cuda:2")
seed = 30
torch.manual_seed(seed)
np.random.seed(seed)
# device


# In[39]:


large = 25; med = 19; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': 20,
          'figure.figsize': (27, 8),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': med}
plt.rcParams.update(params)


# # generate y ( sequence) using reaction difussion model:

# In[40]:


# default num_simulations = 5
num_simulations = 1

# helper function for data visualization
def draw(A,B):
    """draw the concentrations"""
    fig, ax = plt.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')
    

def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4*M
    L += np.roll(M, (0,-1), (0,1)) # right neighbor
    L += np.roll(M, (0,+1), (0,1)) # left neighbor
    L += np.roll(M, (-1,0), (0,1)) # top neighbor
    L += np.roll(M, (+1,0), (0,1)) # bottom neighbor
    
    return L

"""generate the data sequence using reaction diffusion model"""
def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """
    
    # Let's get the discrete Laplacians first
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)
    
    # Now apply the update formula
    diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
    diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t
    
    A += diff_A
    B += diff_B
    
    return A, B, np.concatenate((np.expand_dims(crop_center(A, 32, 32), axis=0), np.expand_dims(crop_center(B, 32, 32),axis=0)), axis=0)



def get_initial_configuration():
    pic = np.load("../data/initial_pic.npy")
    return pic[0], pic[1]

def reaction_diffusion(N, N_simulation_steps, A, B, DA, DB, f, k, delta_t, subsampling_rate):
    """
    The shape of dataArrA and dataArrB should be (timestep, X, Y) where X and Y are the pixels
    """
    seq = []
    for t in range(N_simulation_steps):
        A, B, croppedArr = gray_scott_update(A, B, DA, DB, f, k, delta_t)
        if (t % subsampling_rate == 0):
            seq.append(croppedArr)

    return seq

# crop the image
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_initial_pic(picSize=32, N=100, random_influence=0.1):
    A,B = get_initial_configuration(N, random_influence=random_influence)
    return crop_center(A,picSize,picSize), crop_center(B,picSize,picSize)

# fk_rates: array of feed/kill rates used to generate data, [[fr1,kr1], [fr2,kr2], ...]
# using subsampling to reduce time steps, returning Y with 50 stamps per fk_rate set
def genY(fk_rates, num_simulations = num_simulations, N_simulation_steps = 5000, delta_t = 1.0, grid_size = 100, sampling_rate=100):
    
    dataArr_all = []
    
    for i in range(len(fk_rates)):
        # get feed/kill rates
        f = fk_rates[i][0]
        k = fk_rates[i][1]
        DA = 0.16
        DB = 0.08
        rateArr = []
        for j in range(num_simulations):
            # get initial states
            A, B = get_initial_configuration()
            
            # get datasets
            react_seq = reaction_diffusion(grid_size, N_simulation_steps, A, B, DA, DB, f, k, delta_t, sampling_rate)

#             yArr = np.concatenate((A, B), axis=1)
            rateArr.append(react_seq)
            print("simulating: ", j+1, "/", num_simulations)
        
        dataArr_all.append(rateArr)
        print("total progress: ", i+1, "/", len(fk_rates))

    return np.array(dataArr_all)


# In[41]:


# generate train parameters

# setting f and k
# fix k to 0.62 for now, only change f
# number of different fkABs 
num_train_f = 20
num_train_k = 8

fkAB = []
f_train = np.linspace(0.029, 0.045, num_train_f)
k_train = np.linspace(0.055, 0.062, num_train_k)
xMin_f = f_train.min()
xMax_f = f_train.max()
xMin_k = k_train.min()
xMax_k = k_train.max()
x_all = []
normalized_f_train = ((f_train -xMin_f)/(xMax_f - xMin_f) * 100) + 1
normalized_k_train = ((k_train -xMin_k)/(xMax_k - xMin_k) * 100) + 1
for f_theta in f_train:
    for k_theta in k_train:
        fkAB.append(np.array([f_theta, k_theta]))
for norf in normalized_f_train:
    for nork in normalized_k_train:
        x_all.append(np.array([norf, nork]))
# Diffusion coefficients
# Fixed Diffusion coefficients for now
# DA_train = np.full(num_sets_train, 0.16)
# DB_train = np.full(num_sets_train, 0.08)
# fkAB = np.stack([f_train,k_train], -1)
fkAB = np.array(fkAB)
x_all = np.array(x_all)
# print(x_all.shape)


# In[42]:


# generate validation parameters

# setting f and k
# fix k to 0.62 for now, only change f
# f = np.expand_dims(np.linspace(3.0, 5.8, 10),1)
# k = np.full((10,1), 0.62)
# number of different fkABs 
num_val_f = 5
num_val_k = 4

fkAB_val = []
x_val = []
f_val = np.linspace(0.031, 0.044, num_val_f)
k_val = np.linspace(0.056, 0.061, num_val_k)
normalized_f_val = ((f_val -xMin_f)/(xMax_f - xMin_f) * 100) + 1
normalized_k_val = ((k_val -xMin_k)/(xMax_k - xMin_k) * 100) + 1
for f_theta in f_val:
    for k_theta in k_val:
        fkAB_val.append(np.array([f_theta, k_theta]))
for norf in normalized_f_val:
    for nork in normalized_k_val:
        x_val.append(np.array([norf, nork]))

fkAB_val = np.array(fkAB_val)
x_val = np.array(x_val)
# print(x_val.shape)


# In[43]:


# generate test parameters

# setting f and k
# fix k to 0.62 for now, only change f
# f = np.expand_dims(np.linspace(3.0, 5.8, 10),1)
# k = np.full((10,1), 0.62)
# number of different fkABs 
num_test_f = 5
num_test_k = 4

fkAB_test = []
x_test = []
f_test = np.linspace(0.030, 0.043, num_test_f)
k_test = np.linspace(0.056, 0.060, num_test_k)
# Diffusion coefficients
# Fixed Diffusion coefficients for now

normalized_f_test = ((f_test -xMin_f)/(xMax_f - xMin_f) * 100) + 1
normalized_k_test = ((k_test -xMin_k)/(xMax_k - xMin_k) * 100) + 1
for f_theta in f_test:
    for k_theta in k_test:
        fkAB_test.append(np.array([f_theta, k_theta]))
for norf in normalized_f_test:
    for nork in normalized_k_test:
        x_test.append(np.array([norf, nork]))

fkAB_test = np.array(fkAB_test)
x_test = np.array(x_test)
# print(fkAB_test.shape)


# In[44]:


# fkAB: the parameter set for training
y_train_data = genY(fkAB)
# train data flatten test
# x_all = np.repeat(fkAB, num_simulations, axis = 0)
# boost data to avoid gradient vanishing
# each data point for y with the timestamp info included: 2(A and B) * (32 * 32) <- pixels = 2048
y_all = np.array(y_train_data).reshape(-1, 2048)
y_all = (y_all - y_all.min())/(y_all.max() - y_all.min()) * 100
y_all = y_all.reshape(num_train_f*num_train_k,50,2,32,32)
# take the first 3 steps
y_all = y_all[:,0:6,...]
# print(y_all.shape)


# In[45]:


# print(x_all)
# print(np.array(y_train_data).shape)
# print(np.array(y_all).shape)


# In[46]:


# fkAB_val: the parameter set for validating
y_val = genY(fkAB_val)
# train data flatten test
# x_val = np.repeat(fkAB_val, num_simulations, axis = 0)
# each data point for y with the timestamp info included: 2(A and B) * (30 * 30) <- pixels = 1800
y_val = np.array(y_val).reshape(num_val_f*num_val_k, 50, 2, 32, 32)
y_val = y_val[:,0:6,...]
# print(x_val.shape)
# print(y_val.shape)
y_val = (y_val - y_val.min())/(y_val.max() - y_val.min()) * 100
# print(x_val)


# In[47]:


# fkAB_test: the parameter set for testing
y_test = genY(fkAB_test)
# train data flatten test
# x_test = np.repeat(fkAB_test, num_simulations, axis = 0)
# each data point for y with the timestamp info included: 2(A and B) * (30 * 30) <- pixels = 1800
y_test = np.array(y_test).reshape(num_test_f*num_test_k, 50, 2, 32, 32)
y_test = y_test[:,0:6,...]
# print(x_test)
# print(y_test.shape)
y_test = (y_test - y_test.min())/(y_test.max() - y_test.min()) * 100
# print(y_test)


# In[48]:


# print(x_all)


# In[49]:


# print(np.array(y_all).shape)
# for elem in y_all:
#     for sim in elem:
#         draw(sim[0], sim[1])


# In[50]:


# # replace beta_epsilon_all with fkAB to initiate the mask!
# np.random.seed(3)
# mask_init = np.zeros(len(fkAB))
# mask_init[:5] = 1

# np.random.shuffle(mask_init)
# selected_fkAB = fkAB[mask_init.astype('bool')]
# print(selected_fkAB)
# x_train_init = np.repeat(selected_fkAB, num_simulations,axis =0)

# # use the selected fkAB values to select their corresponding data
# selected_y = np.array(y_train_data)[mask_init.astype('bool')]
# # selected_y.shape[2]*selected_y.shape[3]*selected_y.shape[4] : 50 * 2 * 30 * 30
# # each data point for y with the timestamp info included: 2(A and B) ,(32 , 32) <- pixels 
# y_train_init = selected_y.reshape(selected_y.shape[0]*selected_y.shape[1],selected_y.shape[2],selected_y.shape[3],selected_y.shape[4])
# print(x_train_init.shape, y_train_init.shape)
# # print(mask_init)


# # CNP

# In[51]:


#reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""
    
    def __init__(self, in_dim, out_dim, init_func = torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 64 #16
        self.l2_size = 32 #8
        self.l3_size = 16 #DNE
        
        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, self.l3_size)
        self.l4 = torch.nn.Linear(self.l3_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        self.a3 = torch.nn.Sigmoid()
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
            init_func(self.l4.weight)
        
    def forward(self, inputs):
        return self.l4(self.a3(self.l3(self.a2(self.l2(self.a1(self.l1(inputs)))))))

class ZEncoder(torch.nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.logvar1_size = out_dim
        
        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.logvar1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.logvar1.weight)
        
    def forward(self, inputs):
        

        return self.m1(inputs), self.logvar1(inputs)

"""Original Decoder implementation without convolutional layer"""
# class Decoder(torch.nn.Module):
#     """
#     Takes the x star points, along with a 'function encoding', z, and makes predictions.
#     """
#     def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
#         super(Decoder, self).__init__()
#         self.l1_size = 16 #8
#         self.l2_size = 32 #16
#         self.l3_size = 64 #DNE
        
#         self.l1 = torch.nn.Linear(in_dim, self.l1_size)
#         self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
#         self.l3 = torch.nn.Linear(self.l2_size, self.l3_size)
#         self.l4 = torch.nn.Linear(self.l3_size, out_dim)
        
#         if init_func is not None:
#             init_func(self.l1.weight)
#             init_func(self.l2.weight)
#             init_func(self.l3.weight)
#             init_func(self.l4.weight)
        
#         self.a1 = torch.nn.Sigmoid()
#         self.a2 = torch.nn.Sigmoid()
#         self.a3 = torch.nn.Sigmoid()
        
#     def forward(self, x_pred, z):
#         """x_pred: No. of data points, by x_dim
#         z: No. of samples, by z_dim
#         """
#         zs_reshaped = z.unsqueeze(-1).expand(z.shape[0], x_pred.shape[0]).transpose(0,1)
#         xpred_reshaped = x_pred
        
#         xz = torch.cat([xpred_reshaped, zs_reshaped], dim=1)

#         return self.l4(self.a3(self.l3(self.a2(self.l2(self.a1(self.l1(xz))))).squeeze(-1)))

def MAE(pred, target):
#     print(target.unsqueeze(2).shape)
    loss = torch.abs(pred-(target.unsqueeze(2)[:,1:,...]))
    return loss.mean()


# In[52]:


conv_outDim = 64
init_channels = 4
image_channels_in_encoder = 4
image_channels_in_decoder = 2
kernel_size = 3
lstm_hidden_size = 128

class ConvEncoder(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            #input_shape = (2,30,30)
            nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.Dropout(p = .2),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.output = nn.Sequential(
            #start pushing back
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(p = .2),
            nn.Linear(32, conv_outDim)
        )
        
    def forward(self, x):
        #x.to(torch.float32)
#         print("input shape: ", x.shape)
        #(2,30,30)
        x = self.conv1(x)
       #print(x.shape)
        #(64,124,252)
        x = self.conv2(x)
        #print(x.shape)
        #(64,21,42)
        x = self.conv3(x)
        x = self.conv4(x)
#         print("x before reshape", x.shape)
        x = x.view(x.size(0), -1)
#         print("x after reshape: ", x.shape)
        output = self.output(x)
#         print("shape of encoder output: ", output.shape)
        
        return output


# In[53]:


class ConvDecoder(nn.Module):
    def __init__(self, in_dim, out_dim) :
        super().__init__()
        self.input = nn.Sequential(
            #start pushing back
            nn.Linear(in_dim, conv_outDim),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=conv_outDim, out_channels=init_channels*8, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
#             nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
#             nn.Dropout(p = .2),
#             nn.MaxPool2d(kernel_size = 3)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
#             nn.MaxPool2d(kernel_size = 3)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels_in_decoder, kernel_size = kernel_size,stride = 2, output_padding = 1),
            nn.ReLU()
#             nn.MaxPool2d(kernel_size = 3)
        )

        
    def forward(self, x_pred):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
#         zs_reshaped = z.unsqueeze(-1).expand(z.shape[0], x_pred.shape[0]).transpose(0,1)
#         xpred_reshaped = x_pred
#         print("zs shape: ", zs_reshaped.shape)
#         print("xpred shape: ", xpred_reshaped.shape)
        
#         xz = torch.cat([xpred_reshaped, zs_reshaped], dim=1)
        x = self.input(x_pred)
        x = x.view(-1, conv_outDim, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv4(x)
#         print("predicted image shape", output.shape)
        
        return output


# In[54]:


class DCRNNModel(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, init_func=torch.nn.init.normal_):
        super().__init__()
        self.conv_encoder = ConvEncoder(image_channels_in_encoder)
        self.conv_encoder_in_decoder = ConvEncoder(image_channels_in_decoder)
        self.deconv = ConvDecoder(lstm_hidden_size, y_dim) # (x*, z) -> y*
#         self.lstm_linear = nn.Sequential(
# #             nn.Linear(conv_outDim, conv_reduction_dim+x_dim)
#             nn.Linear(lstm_hidden_size, r_dim),
#             nn.Sigmoid(),
#             nn.Linear(r_dim, r_dim)
#         )
        self.encoder_lstm = nn.LSTM(input_size = conv_outDim+x_dim, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size = conv_outDim+x_dim+z_dim, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
#         self.repr_encoder = REncoder(x_dim+conv_outDim, r_dim) # (x,y)->r
        self.z_encoder = ZEncoder(lstm_hidden_size, z_dim) # r-> mu, logvar
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.xdim = x_dim
        self.y_init = decoder_init
#         self.reduce_dm_decoder = nn.Linear(conv_outDim, conv_reduction_dim)
#         self.reduce_dm_encoder = nn.Linear(conv_outDim, conv_reduction_dim)
        
    # stack x0...xt-1 and x1...xt (seq_len-1, 2,32,32) -> (seq_len-1, 4 ,32,32)
    def stack_y(self, y):
        # x0 -> xt-1
        seq1 = y[:-1]
        # print("x before stacking: ", seq1.shape)
        # x1 -> xt
        seq2 = y[1:]
        seq3 = torch.cat((seq1, seq2), 1)
#         print("y after stacking: ", seq3.shape)
        return seq3
    
    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        rs_all = None
        for i,theta_seq in enumerate(y):
            y_stacked = self.stack_y(theta_seq)
            y_conv_c = self.conv_encoder(y_stacked)
            encode_hidden_state = None
#             print(y_conv_c.shape)
    #         print("shape of y after conv layer: ", y_conv_c.shape)
            # corresponding theta to current y: x[i]
            xy = torch.cat([y_conv_c, x[i].repeat(len(y_stacked)).reshape(-1,x_dim)], dim=1)
#             print("shape of xy: ", xy.shape)
            rs , encode_hidden_state = self.encoder_lstm(xy, encode_hidden_state)
#             self.lstm_linear(rs)
            rs = rs.unsqueeze(0)
#             print("shape of rs: ", rs.shape)
            if rs_all is None:
                rs_all = rs
            else:
                rs_all = torch.vstack((rs_all, rs))
            
#         print("shape of rs_all: ", rs_all.shape)
        r_agg = rs_all.mean(dim=0) # Average over samples
#         print("shape of r_agg: ", r_agg.shape)
        return self.z_encoder(r_agg) # Get mean and variance for q(z|...)
    
    def sample_z(self, mu, logvar,n=1):
        """Reparameterisation trick."""
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_()).to(device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n,z_dim).normal_()).to(device)
        
        # std = torch.exp(0.5 * logvar)
        std = 0.1+ 0.9*torch.sigmoid(logvar)
#         print(mu + std * eps)
        return mu + std * eps

    def KLD_gaussian(self):
        """Analytical KLD between 2 Gaussians."""
        mu_q, logvar_q, mu_p, logvar_p = self.z_mu_all, self.z_logvar_all, self.z_mu_context, self.z_logvar_context

        std_q = 0.1+ 0.9*torch.sigmoid(logvar_q)
        std_p = 0.1+ 0.9*torch.sigmoid(logvar_p)
        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        return torch.distributions.kl_divergence(q, p).sum()
    
    # decoder for the model, need zs(array of [[mean], [var]], length is seq_length)
    def decoder(self, theta, z_mu, z_log, seq_len=5, prev_pred = None):
        # initialize prev_pred
        # if not training, get a initial pic!
        if (prev_pred is None):
            prev_pred = self.y_init
       
        outputs = None
        encoded_states = None
        deconv_encoder_hidden_state = None
#         theta_encoded = self.theta_fc_in_decoder(theta)
#         print("shape of z_mu: ", z_mu.shape)


        
        for i in range(seq_len):
            # encode image to hidden (r)
            convEncoded = self.conv_encoder_in_decoder(prev_pred)
            # conv_outDim -> 1, 4
#             convEncoded = self.reduce_dm_decoder(convEncoded)
    #         print(theta.shape)

            # append theta to every timestamp
            tempTensor = torch.empty(conv_outDim+x_dim+z_dim).to(device)
            tempTensor[:conv_outDim] = convEncoded
#             get the z sample in corresponding timestamp
            tempTensor[conv_outDim:-x_dim] = self.sample_z(z_mu[i], z_log[i])
            tempTensor[-x_dim:] = theta
            
            if encoded_states is None:
                encoded_states = tempTensor
                convEncoded = torch.unsqueeze(tempTensor, 0)
            else:
                encoded_states = torch.vstack((encoded_states, tempTensor))
                convEncoded = encoded_states            
        
    #         print(convEncoded.shape)
            # 4+z_dim+x_dim


            output, deconv_encoder_hidden_state = self.decoder_lstm(convEncoded, deconv_encoder_hidden_state)
    #             output = self.fc_conv_de_to_hidden(output[-1])
            # end of convlstm in decoder
    #         print("shape of output: ", output.shape)



            #start of deconv
    #             output = self.fc_deconv_de_to_hidden(output)
            # final image predicted
            outputs = self.deconv(output)
            outputs = outputs.unsqueeze(1)            
            # update prev_pred to the prediction
            prev_pred = outputs[-1]
#             print("outputs shape: ", outputs.shape)
    #         outputs.append(output)
    #         outputs = torch.stack(outputs, dim=0)
    #         print("shape of final output: ", output.shape)
            
        return outputs
        

    def forward(self, x_t, x_c, y_c, x_ct, y_ct):
        """
        """

        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x_ct, y_ct)
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
#         print("shape of z_mu_all: ", self.z_mu_all.shape)
        outputs = []
        for target in x_t:
            output = self.decoder(target, self.z_mu_all, self.z_logvar_all)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
#         self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)
#         print("shape of zs: ", self.zs.shape)
        return outputs
    


# In[55]:


# all good, no additional modification needed
def random_split_context_target(x,y, n_context):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)

def sample_z(mu, logvar,n=1):
    """Reparameterisation trick."""
    if n == 1:
        eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_())
    else:
        eps = torch.autograd.Variable(logvar.data.new(n,z_dim).normal_())
    
    std = 0.1+ 0.9*torch.sigmoid(logvar)
#     print(mu + std * eps)
    return mu + std * eps

def data_to_z_params(x, y):
    """Helper to batch together some steps of the process."""
    rs_all = None
    for i,theta_seq in enumerate(y):
        y_stacked = dcrnn.stack_y(theta_seq)
        y_conv_c = dcrnn.conv_encoder(y_stacked)
        encode_hidden_state = None
#             print(y_conv_c.shape)
#         print("shape of y after conv layer: ", y_conv_c.shape)
        # corresponding theta to current y: x[i]
        xy = torch.cat([y_conv_c, x[i].repeat(len(y_stacked)).reshape(-1,x_dim)], dim=1)
#             print("shape of xy: ", xy.shape)
        rs , encode_hidden_state = dcrnn.encoder_lstm(xy, encode_hidden_state)
#             self.lstm_linear(rs)
        rs = rs.unsqueeze(0)
#             print("shape of rs: ", rs.shape)
        if rs_all is None:
            rs_all = rs
        else:
            rs_all = torch.vstack((rs_all, rs))

#         print("shape of rs_all: ", rs_all.shape)
    r_agg = rs_all.mean(dim=0) # Average over samples
#         print("shape of r_agg: ", r_agg.shape)
    return dcrnn.z_encoder(r_agg) # Get mean and variance for q(z|...)

def test(x_train, y_train, x_test):
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device))
      
        output_list = None
        for i in range (len(x_test)):
        #           zsamples = sample_z(z_mu, z_logvar) 
            output = dcrnn.decoder(x_test[i:i+1].to(device), z_mu, z_logvar).cpu().unsqueeze(0)
            if output_list is None:
                output_list = output.detach()
            else:
                output_list = torch.vstack((output_list, output.detach()))
    
    return output_list.numpy()


# In[56]:


def train(n_epochs, x_train, y_train, x_val, y_val, x_test, y_test, n_display=500, patience = 5000): #7000, 1000
    train_losses = []
    # mae_losses = []
    # kld_losses = []
    val_losses = []
    test_losses = []

    means_test = []
    stds_test = []
    min_loss = 0. # for early stopping
    wait = 0
    min_loss = float('inf')
    dcrnn.train()
    
    for t in range(n_epochs): 
        opt.zero_grad()
        #Generate data and process
        x_context, y_context, x_target, y_target = random_split_context_target(
                                x_train, y_train, int(len(y_train)*0.2)) #0.25, 0.5, 0.05,0.015, 0.01
#         print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)    

        # for overfitting use val as context, for actual training, use code above to split context!
#         x_context = x_train
#         y_context = y_train
#         x_target  = x_train
#         y_target  = y_train

        x_c = torch.from_numpy(x_context).float().to(device)
        x_t = torch.from_numpy(x_target).float().to(device)
        y_c = torch.from_numpy(y_context).float().to(device)
        y_t = torch.from_numpy(y_target).float().to(device)

        x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
        y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

        y_pred = dcrnn(x_t, x_c, y_c, x_ct, y_ct)
        
#         print("shape of y_pred: ", y_pred.shape)
#         print("shape of y_t: ", y_t.shape)

        train_loss = MAE(y_pred, y_t) + dcrnn.KLD_gaussian()
        mae_loss = MAE(y_pred, y_t)
        kld_loss = dcrnn.KLD_gaussian()
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(dcrnn.parameters(), 5) #10
        opt.step()
        
        #val loss
        y_val_pred = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_val).float())
#         print("shape of y_val_pred: ", y_val_pred.shape)
#         print("shape of y_val: ", y_val.shape)
        val_loss = MAE(torch.from_numpy(y_val_pred).float(),torch.from_numpy(y_val).float())
        #test loss
        y_test_pred = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_test).float())
        test_loss = MAE(torch.from_numpy(y_test_pred).float(),torch.from_numpy(y_test).float())

        if t % n_display ==0:
            print('train loss:', train_loss.item(), 'mae:', mae_loss.item(), 'kld:', kld_loss.item(), flush=True)
            print('val loss:', val_loss.item(), 'test loss:', test_loss.item(), flush=True)
            ypred_allset.append(y_pred)
#             print(y_train)

        if t % (n_display/10) ==0:
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            test_losses.append(test_loss.item())
#             mae_losses.append(mae_loss.item())
#             kld_losses.append(kld_loss.item())
        
#         if train_loss.item() < 10:
#             return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all
        
#         #early stopping
        if val_loss < min_loss:
            wait = 0
            min_loss = val_loss
            
        elif val_loss >= min_loss:
            wait += 1
            if wait == patience:
                print('Early stopping at epoch: %d' % t)
                return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all
        
    return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all


# In[57]:


# pass in fkAB as beta_epsilon_all
def select_data(x_train, y_train, beta_epsilon_all, yall_set, score_array, selected_mask):

    # make sure it does not select selected element
    # mask: [0,1,1,0,0,1]
    # 1- 2*mask: [1,-1,-1,1,1,-1]
    mask_score_array = score_array*(1-(selected_mask*2))
    # print('mask_score_array',mask_score_array)
    select_index = np.argmax(mask_score_array)
    print('select_index:',select_index)


    selected_x = beta_epsilon_all[select_index:select_index+1]
    selected_y = yall_set[select_index]

    x_train1 = np.repeat(selected_x,num_simulations,axis =0)
    x_train = np.concatenate([x_train, x_train1],0)
    
    y_train1 = selected_y
    print("selected y shape: ", y_train1.shape)
    y_train = np.concatenate([y_train, y_train1],0)
    print("y after concatenation: ", y_train.shape)
 
    selected_mask[select_index] = 1
    
    return x_train, y_train, selected_mask



# In[58]:


# pass in fkAB as beta_epsilon_all
def calculate_score(x_train, y_train, beta_epsilon_all):
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    # query z_mu, z_var of the current training data
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device))

        score_list = []
        for i in range(len(beta_epsilon_all)):
            # generate x_search
            x1 = beta_epsilon_all[i:i+1]
            x_search = np.repeat(x1,num_simulations,axis =0)
            x_search = torch.from_numpy(x_search).float()

            # generate y_search based on z_mu, z_var of current training data
            output_list = []
            for j in range (len(x_search)):
                zsamples = sample_z(z_mu, z_logvar) 
                output = dcrnn.decoder(x_search[j:j+1].to(device), zsamples).cpu()
                output_list.append(output.detach().numpy())

            y_search = np.concatenate(output_list)
            y_search = torch.from_numpy(y_search).float()

            x_search_all = torch.cat([x_train,x_search],dim=0)
            y_search_all = torch.cat([y_train,y_search],dim=0)

            # generate z_mu_search, z_var_search
            z_mu_search, z_logvar_search = data_to_z_params(x_search_all.to(device),y_search_all.to(device))
            
            # calculate and save kld
            mu_q, var_q, mu_p, var_p = z_mu_search,  0.1+ 0.9*torch.sigmoid(z_logvar_search), z_mu, 0.1+ 0.9*torch.sigmoid(z_logvar)

            std_q = torch.sqrt(var_q)
            std_p = torch.sqrt(var_p)

            p = torch.distributions.Normal(mu_p, std_p)
            q = torch.distributions.Normal(mu_q, std_q)
            score = torch.distributions.kl_divergence(p, q).sum()

            score_list.append(score.item())

        score_array = np.array(score_list)

    return score_array




# BO search:

# In[59]:


# TODO: replace np.linespace with our correct ones for reaction diffusion data
def mae_plot(mae, selected_mask):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    selected_mask = selected_mask.reshape(30,9)
    mae_min, mae_max = 0, 1200

    fig, ax = plt.subplots(figsize=(16, 5))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, mae, cmap='binary', vmin=mae_min, vmax=mae_max)
    ax.set_title('MAE Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.show()

def score_plot(score, selected_mask):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    score_min, score_max = 0, 1
    selected_mask = selected_mask.reshape(30,9)
    score = score.reshape(30,9)
    fig, ax = plt.subplots(figsize=(16, 5))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, score, cmap='binary', vmin=score_min, vmax=score_max)
    ax.set_title('Score Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.show()




# In[60]:


def MAE_MX(y_pred, y_test):
    y_pred = y_pred.reshape(2, 2, 2, 32, 32)
    y_test = y_test.reshape(2, 2, 2, 32, 32)
    mae_matrix = np.mean(np.abs(y_pred - y_test),axis=(2,3,4))
    mae = np.mean(np.abs(y_pred - y_test))
    return mae_matrix, mae


# In[61]:


# beta = np.repeat(np.expand_dims(np.linspace(1.1, 4., 30),1),9,1)
# epsilon = np.repeat(np.expand_dims(np.linspace(0.25, 0.65, 9),0),30,0)
# beta_epsilon = np.stack([beta,epsilon],-1)
# beta_epsilon_flatten_test = beta_epsilon.reshape(-1,2)

# ytest_set, ytest_mean, ytest_std = seir(num_days,beta_epsilon_flatten_test,num_simulations)
# print(beta_epsilon_flatten_test.shape)
# print(ytest_set.shape)
# print(ytest_mean.shape)
# print(ytest_std.shape)

# x_test = np.repeat(beta_epsilon_flatten_test,num_simulations,axis =0)
# y_test = ytest_set.reshape(-1,100)
# print(x_test.shape, y_test.shape)


# In[62]:


r_dim = 64
z_dim = 64 #8
x_dim = 2 #
y_dim = 2 
# N = 100000 #population


# In[63]:


ypred_allset = []
ypred_testset = []
mae_allset = []
maemetrix_allset = []
mae_testset = []
score_set = []
mask_set = []

#number of parameters
decoder_init = torch.unsqueeze(torch.from_numpy(np.load("../data/initial_pic.npy")).float().to(device), 0)
print(decoder_init.shape)
dcrnn = DCRNNModel(x_dim, y_dim, r_dim, z_dim).to(device)
opt = torch.optim.Adam(dcrnn.parameters(), 1e-3) #1e-3
pytorch_total_params = sum(p.numel() for p in dcrnn.parameters() if p.requires_grad)
print(pytorch_total_params)


# In[29]:


# offline model
ypred_allset = []
dcrnn.train()
train_losses, val_losses, test_losses, z_mu, z_logvar = train(20000,x_all,y_all,x_val, y_val, x_test, y_test,500, 1500) #20000, 5000


# In[ ]:


plt.plot(train_losses)
plt.title('Loss Graph')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('train_loss_graph_20by8_theta_stacked_y_stnp_64_rdim_2_theta.png')


# In[ ]:


plt.plot(val_losses)
plt.title('Validation Graph')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('val_loss_graph_20by8_theta_stacked_y_stnp_64_rdim_2_theta.png')


# In[ ]:


# np.save("np\ypred_allset_50_theta_baseline.npy", torch.stack(ypred_allset).cpu().detach().numpy())
np.save("train_loss_20by8_theta_stacked_y_stnp_64_rdim_2_theta_%d.npy" % seed, np.array(train_losses))
np.save("val_loss_20by8_theta_stacked_y_stnp_64_rdim_2_theta_%d.npy" % seed, np.array(val_losses))
np.save("test_loss_20by8_theta_stacked_y_stnp_64_rdim_2_theta_%d.npy" % seed, np.array(test_losses))
torch.save(dcrnn.state_dict(), "5_seq_20by8_theta_stacked_y_stnp_64_rdim_2_theta_%d.pt" % seed)
print('training finished, dicts saved')


# In[60]:


# dcrnn.load_state_dict(torch.load("state_dicts/stnp_5_seq_50_theta_2_lstm.pt"))


# In[57]:


# print(len(ypred_allset))
# for seq in ypred_allset[2]:
#     for img in seq:
#         draw(img[0].cpu().detach().numpy(), img[1].cpu().detach().numpy())


# In[30]:


# ypred_allarr = np.stack(ypred_allset,0)
# ypred_testarr = np.stack(ypred_testset,0) 
# maemetrix_allarr = np.stack(maemetrix_allset,0) 
# mae_allarr = np.stack(mae_allset,0)
# mae_testarr = np.stack(mae_testset,0)
# score_arr = np.stack(score_set,0)
# mask_arr = np.stack(mask_set,0)


# In[31]:


# print(ypred_allarr.shape)
# draw_yarr = np.squeeze(ypred_allarr, axis=0)
# draw_yarr = draw_yarr.reshape(-1, 2, 32, 32)
# print(draw_yarr.shape)
# for elem in draw_yarr:
#     draw(elem[0], elem[1])


# In[32]:


# vis_pred = test(torch.from_numpy(x_all).float().to(device), torch.from_numpy(y_all).float().to(device), torch.from_numpy(x_all).float().to(device))

# vis_pred=[]
# for i in range(5):
#     vis_pred.append(vis_res(torch.from_numpy(x_all).float().to(device), torch.from_numpy(y_all).float().to(device), torch.from_numpy(testArr).float().to(device)).reshape(-1,2,32,32))


# In[33]:


# print(y_all.shape)


# In[34]:


# # print(vis_pred_shaped.shape)
# for i,theta in enumerate(vis_pred):
#     for j,seq in enumerate(theta):
#         draw_ground(seq[0][0], seq[0][1], y_all[i][j+1][0], y_all[i][j+1][1])
# #       draw(img[0][0], img[0][1])

# vis_pred = np.array(vis_pred)
# print("max diff: ", np.max(vis_pred[1:] - vis_pred[:-1]))
# print("min diff: ", np.min(vis_pred[1:] - vis_pred[:-1]))
# print(vis_pred[1:] - vis_pred[:-1])


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')
# %cd '/content/drive/MyDrive/iclr2022_paper1/2D_NP_LIG_heldout/'


# In[ ]:




