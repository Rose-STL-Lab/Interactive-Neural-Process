#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[9]:


device = torch.device("cuda:1")
seed = 31
torch.manual_seed(seed)
np.random.seed(seed)
# device


# In[10]:


large = 25; med = 19; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': 20,
          'figure.figsize': (27, 8),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': med}
plt.rcParams.update(params)


# # Load Data:

# In[94]:


x_all = np.load("../data/x_all.npy")
x_val = np.load("../data/x_val.npy")
x_test = np.load("../data/x_test.npy")
y_all = np.load("../data/y_all.npy")
y_val = np.load("../data/y_val.npy")
y_test = np.load("../data/y_test.npy")
# print(y_test)


# In[12]:


# print(x_all)


# In[13]:


# print(np.array(y_all).shape)
# for elem in y_all:
#     for sim in elem:
#         draw(sim[0], sim[1])


# In[14]:


# # replace beta_epsilon_all with fkAB to initiate the mask!
# mask_init = np.zeros(len(x_all))
# mask_init[:8] = 1

# np.random.shuffle(mask_init)
# x_train_init = x_all[mask_init.astype('bool')]
# print(x_train_init)

# # use the selected fkAB values to select their corresponding data
# y_train_init = np.array(y_all)[mask_init.astype('bool')]
# # selected_y.shape[2]*selected_y.shape[3]*selected_y.shape[4] : 50 * 2 * 30 * 30
# # each data point for y with the timestamp info included: 2(A and B) ,(32 , 32) <- pixels 
# print(x_train_init.shape, y_train_init.shape)
# print(mask_init)


# # CNP

# In[15]:


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


# In[16]:


conv_outDim = 64
init_channels = 4
image_channels_in_encoder = 4
image_channels_in_decoder = 2
kernel_size = 3
lstm_hidden_size = 128
decoder_init = torch.unsqueeze(torch.from_numpy(np.load("../data/initial_pic.npy")).float().to(device), 0)

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


# In[17]:


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


# In[18]:


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
    


# In[19]:


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

def data_to_z_params(x, y, calc_score = False):
    """Helper to batch together some steps of the process."""
    rs_all = None
    for i,theta_seq in enumerate(y):
        if calc_score:
            theta_seq = torch.cat([decoder_init, theta_seq])
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


# In[84]:


def train(n_epochs, x_train, y_train, x_val, y_val, x_test, y_test, n_display=500, patience = 5000): #7000, 1000
    train_losses = []
    mae_losses = []
    kld_losses = []
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


# In[85]:


# # pass in fkAB as beta_epsilon_all
# def select_data(x_train, y_train, beta_epsilon_all, yall_set, score_array, selected_mask, select_size):

#     # make sure it does not select selected element
#     # mask: [0,1,1,0,0,1]
#     # 1- 2*mask: [1,-1,-1,1,1,-1]
#     mask_score_array = score_array*(1-selected_mask)
#     # print('mask_score_array',mask_score_array)
#     select_index = np.argpartition(mask_score_array, -select_size)[-select_size:]
#     print('select_index:',select_index)

#     # beta_epsilon_all: x_all
#     selected_x = beta_epsilon_all[select_index]
#     selected_y = yall_set[select_index]

#     x_train = np.concatenate([x_train, selected_x],0)
    
#     y_train1 = selected_y
#     print("selected y shape: ", y_train1.shape)
#     y_train = np.concatenate([y_train, y_train1],0)
#     print("y after concatenation: ", y_train.shape)
 
#     selected_mask[select_index] = 1
    
#     return x_train, y_train, selected_mask

# takes in x and y data, generate two sets, one is x and y in batch, the other is x and y removed from all data
# returns selected x, selected y, rest of dataset x, rest of dataset y
def generate_batch(x,y, batch_size):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=batch_size, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)


# In[86]:


# initialize search_data_x to the dataset with x_init removed
def calculate_score(x_train, y_train, x_search):
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_search = torch.from_numpy(x_search).float()
    dcrnn.eval()

    # query z_mu, z_var of the current training data
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device))
        
        output_list = []
        for theta in x_search:
            output = dcrnn.decoder(theta, z_mu, z_logvar)
#             print("shape of output in calculating score: ", output.shape)
            output_list.append(output)
        outputs = torch.stack(output_list, dim=0)

        y_search = outputs.squeeze(2)
#         print("shape of y_search: ", y_search.shape)
#         print("shape of y_train: ", y_train.shape)
#         print("shape of x_train: ", x_train.shape)
#         print("shape of x_search: ", x_search.shape)

        x_search_all = torch.cat([x_train.to(device),x_search.to(device)],dim=0)
        y_search_all = torch.cat([y_train[:,1:,...].to(device),y_search],dim=0)
        
#         print("shape of y_search_all: ", y_search_all.shape)
#         print("shape of x_search_all: ", x_search_all.shape)

        # generate z_mu_search, z_var_search
        z_mu_search, z_logvar_search = data_to_z_params(x_search_all.to(device),y_search_all.to(device), calc_score = True)

        # calculate and save kld
        mu_q, var_q, mu_p, var_p = z_mu_search,  0.1+ 0.9*torch.sigmoid(z_logvar_search), z_mu, 0.1+ 0.9*torch.sigmoid(z_logvar)

        std_q = torch.sqrt(var_q)
        std_p = torch.sqrt(var_p)

        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        score = torch.distributions.kl_divergence(q, p).sum()


    return score




# BO search:

# In[87]:


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




# In[88]:


def MAE_MX(y_pred, y_test):
    y_pred = y_pred.reshape(160, 5, 2, 32, 32)
    y_test = y_test[:,1:,...].reshape(160, 5, 2, 32, 32)
    mae_matrix = np.mean(np.abs(y_pred - y_test),axis=(2,3,4))
    mae = np.mean(np.abs(y_pred - y_test))
    return mae_matrix, mae


# In[89]:


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


# In[90]:


r_dim = 64
z_dim = 64 #8
x_dim = 2 #
y_dim = 2 
# N = 100000 #population


# # Offline

# In[91]:


# ypred_allset = []
# ypred_testset = []
# mae_allset = []
# maemetrix_allset = []
# mae_testset = []
# score_set = []
# mask_set = []

# #number of parameters
# decoder_init = torch.unsqueeze(torch.from_numpy(np.load("initial_pic.npy")).float().to(device), 0)
# print(decoder_init.shape)
# dcrnn = DCRNNModel(x_dim, y_dim, r_dim, z_dim).to(device)
# opt = torch.optim.Adam(dcrnn.parameters(), 1e-3) #1e-3
# pytorch_total_params = sum(p.numel() for p in dcrnn.parameters() if p.requires_grad)
# print(pytorch_total_params)


# # Active Learning

# In[92]:


ypred_allset = []
ypred_testset = []
mae_allset = []
maemetrix_allset = []
mae_testset = []
score_set = []
# save the value for all y_train
yall_set = np.array(y_all)
print(yall_set.shape)

decoder_init = torch.unsqueeze(torch.from_numpy(np.load("../data/initial_pic.npy")).float().to(device), 0)
dcrnn = DCRNNModel(x_dim, y_dim, r_dim, z_dim).to(device)
opt = torch.optim.Adam(dcrnn.parameters(), 1e-3) #1e-3

y_pred_test_list = []
y_pred_all_list = []
all_mae_matrix_list = []
all_mae_list = []
test_mae_list = []
score_list = []

# get initial choices of data
# batch size is 8
x_train,y_train, search_data_x, search_data_y = generate_batch(x_all, y_all, 5)


for i in range(10): #8
    dcrnn.train()
    print('training data shape:', x_train.shape, y_train.shape, flush=True)

    train_losses, val_losses, test_losses, z_mu, z_logvar = train(20000,x_train,y_train,x_val, y_val, x_test, y_test,500, 1500) #20000, 5000
    y_pred_test = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_test).float())
#     print("y_pred_shape: ", y_pred_test.shape)
    y_pred_test_list.append(y_pred_test)


    test_mae = MAE(torch.from_numpy(y_pred_test).float(),torch.from_numpy(y_test).float())
    test_mae_list.append(test_mae.item())
    print('Test MAE:',test_mae.item(), flush=True)

    y_pred_all = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_all).float())
#     print("shape of y_pred_all: ", y_pred_all.shape)
    y_pred_all_list.append(y_pred_all)
    mae_matrix, mae = MAE_MX(y_pred_all, y_all)


    all_mae_matrix_list.append(mae_matrix)
    all_mae_list.append(mae)
    print('All MAE:',mae, flush=True)

    
    reward_list = []
    index_list = []
    for k in range(int(len(search_data_x))):
        index = np.random.choice(len(search_data_x), 5, replace=False).tolist()
        index_list.append(index)
        search_data_x_batch = np.stack([search_data_x[i] for i in index],0)
#         print("shape of search_data_y_all: ", search_data_y_all.shape)
        reward = calculate_score(x_train, y_train, search_data_x_batch)
        reward_list.append(reward.item())
    
    # np.save('seed%d_reward_list_itr%d.npy' % (seed, i+1),np.array(reward_list))
    # np.save('seed%d_index_list_itr%d.npy' % (seed, i+1),np.stack(index_list))
    # torch.save(dcrnn.state_dict(), '5_seq_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_seed%d_itr%d.pt' % (seed, i+1))
    # np.save('seed%d_test_mae_list_itr%d.npy' % (seed, i+1),np.stack(test_mae_list))    
    # np.save('seed%d_all_mae_matrix_list_itr%d.npy' % (seed, i+1),np.stack(all_mae_matrix_list))    
    # np.save('seed%d_all_mae_list_itr%d.npy' % (seed, i+1),np.stack(all_mae_list))  

#     print('reward_list:',reward_list)
    selected_ind = np.argmax(np.array(reward_list))
    x_train = np.concatenate((x_train, [search_data_x[i] for i in index_list[selected_ind]]), axis=0)
    y_train = np.concatenate((y_train, [search_data_y[i] for i in index_list[selected_ind]]), axis=0)
    
    search_data_x = [e for i, e in enumerate(search_data_x) if i not in index_list[selected_ind]]
    search_data_y = [e for i, e in enumerate(search_data_y) if i not in index_list[selected_ind]]
    print('remained scenarios:', len(search_data_x), flush=True)    


y_pred_all_arr = np.stack(y_pred_all_list,0)
y_pred_test_arr = np.stack(y_pred_test_list,0)
all_mae_matrix_arr = np.stack(all_mae_matrix_list,0)
all_mae_arr = np.stack(all_mae_list,0)
test_mae_arr = np.stack(test_mae_list,0)
# score_arr = np.stack(score_list,0)

ypred_allset.append(y_pred_all_arr)
ypred_testset.append(y_pred_test_arr)
maemetrix_allset.append(all_mae_matrix_arr)
mae_allset.append(all_mae_arr)
mae_testset.append(test_mae_arr)
# score_set.append(score_arr)

# ypred_allarr = np.stack(ypred_allset,0)
# ypred_testarr = np.stack(ypred_testset,0) 
# maemetrix_allarr = np.stack(maemetrix_allset,0) 
# mae_allarr = np.stack(mae_allset,0)
# mae_testarr = np.stack(mae_testset,0)
# score_arr = np.stack(score_set,0)
# mask_arr = np.stack(mask_set,0)


# In[65]:


# np.save("np\ypred_allset_50_theta_baseline.npy", torch.stack(ypred_allset).cpu().detach().numpy())
# np.save("ypred_all_final_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning.npy", np.array(ypred_allset))
np.save("final_ypred_testset_final_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(ypred_testset))
np.save("final_maemetrix_allset_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(maemetrix_allset))
np.save("final_mae_allset_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(mae_allset))
np.save("final_mae_testset_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.npy" % seed, np.array(mae_testset))
torch.save(dcrnn.state_dict(), "final_5_seq_20by8_theta_stacked_y_stnp_64_rdim_2_theta_active_learning_%d.pt" % seed)
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




