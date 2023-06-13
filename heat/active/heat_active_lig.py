#!/usr/bin/env python
# coding: utf-8


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
from collections import defaultdict

# In[25]:


device = torch.device("cuda:2")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# device


# In[26]:


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

# In[27]:


x_all = np.load("../data/x_train.npy")
x_val = np.load("../data/x_val.npy")
x_test = np.load("../data/x_test.npy")
y_all = np.load("../data/y_train.npy").reshape(384,32,1,32)
y_val = np.load("../data/y_val.npy").reshape(64,32,1,32)
y_test = np.load("../data/y_test.npy").reshape(64,32,1,32)
y_all = y_all[:,::8,...]
y_val = y_val[:,::8,...]
y_test = y_test[:,::8,...]
initial = np.load("../data/initial_pic.npy").reshape(1,32)
# print(y_test)





print(y_all.shape)
print(y_val.shape)
print(y_test.shape)


from datetime import datetime


#reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""
    
    def __init__(self, in_dim, out_dim, init_func = torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 64 
        self.l2_size = 32
        self.l3_size = 16 
        
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

def MAE(pred, target):
    loss = torch.abs(pred-target.unsqueeze(2))
    return loss.mean()


# In[32]:


conv_outDim = 64
init_channels = 4
image_channels_in_encoder = 2
image_channels_in_decoder = 1
kernel_size = 3
lstm_hidden_size = 64
decoder_init = torch.from_numpy(initial).float().to(device)

class ConvEncoder(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=image_channels, out_channels=init_channels, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels, out_channels=init_channels*2, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size = kernel_size,stride = 2),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(32, conv_outDim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if (x.ndim == 2):
            x = x.view(x.size(0))
        else:
            x = x.view(x.size(0), -1)
        output = self.output(x)
        
        return output


# In[33]:


class ConvDecoder(nn.Module):
    def __init__(self, in_dim, out_dim) :
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, conv_outDim),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=conv_outDim, out_channels=init_channels*8, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size = kernel_size,stride = 2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=init_channels*2, out_channels=image_channels_in_decoder, kernel_size = kernel_size,stride = 2, output_padding = 1),
            nn.ReLU()
        )

        
    def forward(self, x_pred):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
        x = self.input(x_pred)
        x = x.view(-1, conv_outDim, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv4(x)
#         print("predicted image shape", output.shape)
        
        return output


# In[34]:


class DCRNNModel(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, init_func=torch.nn.init.normal_):
        super().__init__()
        self.conv_encoder = ConvEncoder(image_channels_in_encoder)
        self.conv_encoder_in_decoder = ConvEncoder(image_channels_in_decoder)
        self.deconv = ConvDecoder(lstm_hidden_size, y_dim) # (x*, z) -> y*
        self.encoder_lstm = nn.LSTM(input_size = conv_outDim+x_dim, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size = conv_outDim+x_dim+z_dim, hidden_size = lstm_hidden_size, num_layers = 1, batch_first=True)
        self.z_encoder = ZEncoder(lstm_hidden_size, z_dim) # r-> mu, logvar
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.xdim = x_dim
        self.y_init = decoder_init
        
    # stack x0...xt-1 and x1...xt (seq_len-1, 2,32,32) -> (seq_len-1, 4 ,32,32)
    def stack_y(self, y):
#         print("shape of y:", y.shape)
        # x0 -> xt-1
        seq1 = np.insert(y.cpu(), 0, initial, axis = 0)[:-1].to(device)
#         print("y before stacking: ", seq1.shape)
        # x1 -> xt
        seq3 = torch.cat((seq1, y), 1)
#         print("y after stacking: ", seq3.shape)
        return seq3
    
    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        rs_all = None
        for i,theta_seq in enumerate(y):
#             print("shape of data: ", y.shape)
            y_stacked = self.stack_y(theta_seq)
#             print(y_stacked.shape)
            y_conv_c = self.conv_encoder(y_stacked)
            encode_hidden_state = None
#             print(y_conv_c.shape)
    #         print("shape of y after conv layer: ", y_conv_c.shape)
            # corresponding theta to current y: x[i]
            xy = torch.cat([y_conv_c, x[i].repeat(len(y_stacked)).reshape(-1,x_dim)], dim=1)
#             print("shape of xy: ", xy.shape)
            rs , encode_hidden_state = self.encoder_lstm(xy, encode_hidden_state)
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
    def decoder(self, theta, z_mu, z_log, seq_len=4, prev_pred = None):
        # initialize prev_pred
        # if not training, get a initial pic!
        if (prev_pred is None):
            prev_pred = self.y_init
       
        outputs = None
        encoded_states = None
        deconv_encoder_hidden_state = None
#         print("shape of z_mu: ", z_mu.shape)


        
        for i in range(seq_len):
            # encode image to hidden (r)
            convEncoded = self.conv_encoder_in_decoder(prev_pred)
            # conv_outDim -> 1, 4
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
        
#             print(convEncoded.shape)
            # 4+z_dim+x_dim


            output, deconv_encoder_hidden_state = self.decoder_lstm(convEncoded, deconv_encoder_hidden_state)
            # end of convlstm in decoder
    #         print("shape of output: ", output.shape)



            #start of deconv
            # final image predicted
            outputs = self.deconv(output)
            outputs = outputs.unsqueeze(1)            
            # update prev_pred to the prediction
            prev_pred = outputs[-1]
#             print("outputs shape: ", outputs.shape)
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
        return outputs
    


# In[35]:


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
        y_stacked = dcrnn.stack_y(theta_seq)
        y_conv_c = dcrnn.conv_encoder(y_stacked)
        encode_hidden_state = None
#             print(y_conv_c.shape)
#         print("shape of y after conv layer: ", y_conv_c.shape)
        # corresponding theta to current y: x[i]
        xy = torch.cat([y_conv_c, x[i].repeat(len(y_stacked)).reshape(-1,x_dim)], dim=1)
#             print("shape of xy: ", xy.shape)
        rs , encode_hidden_state = dcrnn.encoder_lstm(xy, encode_hidden_state)
        rs = rs.unsqueeze(0)
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
            output = dcrnn.decoder(x_test[i:i+1].to(device), z_mu, z_logvar).cpu().unsqueeze(0)
            if output_list is None:
                output_list = output.detach()
            else:
                output_list = torch.vstack((output_list, output.detach()))
    
    return output_list.numpy()


# In[36]:


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
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time, flush=True)
            
            ypred_allset.append(y_pred)
#             print(y_train)

        if t % (n_display/10) ==0:
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            test_losses.append(test_loss.item())
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


# In[74]:


r_dim = 64
z_dim = 64 #8
x_dim = 3 #
y_dim = 1
# N = 100000 #population

ypred_allset = []
ypred_testset = []
mae_allset = []
maemetrix_allset = []
mae_testset = []
score_set = []
mask_set = []
y_pred_test_list = []
test_mae_list = []
y_pred_all_list = []
#number of parameters
dcrnn = DCRNNModel(x_dim, y_dim, r_dim, z_dim).to(device)
opt = torch.optim.Adam(dcrnn.parameters(), 1e-3) #1e-3
pytorch_total_params = sum(p.numel() for p in dcrnn.parameters() if p.requires_grad)
print(pytorch_total_params)






# In[44]:


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
        y_search_all = torch.cat([y_train.to(device),y_search],dim=0)
        
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



# In[45]:


x_train = np.load('x_train_initial.npy')[:5]
y_train = np.load('y_train_initial.npy')[:5]
search_data_x = np.load('search_data_x_initial.npy')
search_data_y = np.load('search_data_y_initial.npy')

for i in range(9): #8
    dcrnn.train()
    print('training data shape:', x_train.shape, y_train.shape, flush=True)

    train_losses, val_losses, test_losses, z_mu, z_logvar = train(5000,x_train,y_train,x_val, y_val, x_test, y_test,500, 1000) #20000, 5000
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

    
    reward_list = []
    index_list = []
    for k in range(int(len(search_data_x))):
        index = np.random.choice(len(search_data_x), 5, replace=False).tolist()
        index_list.append(index)
        search_data_x_batch = np.stack([search_data_x[i] for i in index],0)
#         print("shape of search_data_y_all: ", search_data_y_all.shape)
        reward = calculate_score(x_train, y_train, search_data_x_batch)
        reward_list.append(reward.item())  

#     print('reward_list:',reward_list)
    selected_ind = np.argmax(np.array(reward_list))
    x_train = np.concatenate((x_train, [search_data_x[i] for i in index_list[selected_ind]]), axis=0)
    y_train = np.concatenate((y_train, [search_data_y[i] for i in index_list[selected_ind]]), axis=0)
    
    search_data_x = [e for i, e in enumerate(search_data_x) if i not in index_list[selected_ind]]
    search_data_y = [e for i, e in enumerate(search_data_y) if i not in index_list[selected_ind]]
    print('remained scenarios:', len(search_data_x), flush=True)    


y_pred_all_arr = np.stack(y_pred_all_list,0)
y_pred_test_arr = np.stack(y_pred_test_list,0)
test_mae_arr = np.stack(test_mae_list,0)

ypred_allset.append(y_pred_all_arr)
ypred_testset.append(y_pred_test_arr)
mae_testset.append(test_mae_arr)

np.save("ypred_testset_%d" % seed, np.array(ypred_testset))
np.save("mae_testset_%d" % seed, np.array(mae_testset))
torch.save(dcrnn.state_dict(), "lig_seed%d_final.pt" % seed)
print("training done, all results saved")


