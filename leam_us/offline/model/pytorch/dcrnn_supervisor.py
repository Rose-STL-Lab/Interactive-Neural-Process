import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import mae_loss
from model.pytorch.loss import mae_metric
from model.pytorch.loss import rmse_metric
from model.pytorch.loss import kld_gaussian_loss
# from model.pytorch.loss import meanstd
import csv
# from model.pytorch.loss import std_diff
# from model.pytorch.loss import mean_std
# from model.pytorch.loss import l2_mean_std_loss
# from torch import autograd

device = torch.device("cuda:1")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DCRNNSupervisor:
    def __init__(self, random_seed, iteration, max_itr, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.iteration = iteration
        self.max_itr = max_itr

        # logging.
        self._log_dir = self._get_log_dir(kwargs, self.random_seed, self.iteration)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        # self._data = kwargs.get('data_itr')
        self._data = utils.load_dataset(**self._data_kwargs)

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.input0_dim = int(self._model_kwargs.get('input0_dim', 24))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda(device) if torch.cuda.is_available() else dcrnn_model
        self.z_mean_all=None
        self.z_var_temp_all=None
        self.num_batches = None #int(0)
        self.batch_size = int(self._data_kwargs.get('batch_size'))
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs, random_seed, iteration):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'

            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s_%d_%d/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'), random_seed, iteration)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch, z_total, outputs, saved_model, val_mae, mae, rmse):
        if not os.path.exists('seed%d/itr%d' % (self.random_seed, self.iteration)):
            os.makedirs('seed%d/itr%d' % (self.random_seed, self.iteration))

        config = dict(self._kwargs)
        config['model_state_dict'] = saved_model
        config['epoch'] = epoch
        torch.save(config, 'seed%d/itr%d/model_epo%d.tar' % (self.random_seed, self.iteration, epoch))
        torch.save(z_total, 'seed%d/itr%d/z_epo%d.tar' % (self.random_seed, self.iteration, epoch))
        np.savez_compressed('seed%d/itr%d/test_epo%d.npz'% (self.random_seed, self.iteration, epoch), **outputs)
        self._logger.info("Saved model at {}".format(epoch))

        with open(r'metric_seed%d.csv'% (self.random_seed), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([val_mae, mae, rmse])

        return 'seed%d/itr%d/model_epo%d.tar' % (self.random_seed, self.iteration, epoch)


    def load_model(self):
        assert os.path.exists('seed%d/itr%d/z_epo%d.tar' % (self.random_seed, self.iteration, self._epoch_num)), 'Z at epoch %d not found' % self._epoch_num
        checkpoint1 = torch.load('seed%d/itr%d/z_epo%d.tar' % (self.random_seed, self.iteration, self._epoch_num), map_location='cpu')
        self.z_mean_all = checkpoint1[0].to(device)
        self.z_var_temp_all = checkpoint1[1].to(device)
        self._setup_graph()
        assert os.path.exists('seed%d/itr%d/model_epo%d.tar' % (self.random_seed, self.iteration, self._epoch_num)), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('seed%d/itr%d/model_epo%d.tar' % (self.random_seed, self.iteration, self._epoch_num), map_location='cpu')

        pretrained_dict = checkpoint['model_state_dict']
        model_dict = self.dcrnn_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.dcrnn_model.load_state_dict(model_dict)
 
        self._logger.info("Loaded model at {}".format(self._epoch_num))
        
    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y, x0) in enumerate(val_iterator):
                x, y, x0= self._prepare_data(x, y, x0)
                output,_ = self.dcrnn_model(x, y, x0, test=True, z_mean_all=self.z_mean_all, z_var_temp_all=self.z_var_temp_all)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0,z_mean_all=None, z_var_temp_all=None):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()

            y_truths = []
            y_preds = []

            for _, (x, y, x0) in enumerate(val_iterator):
                x, y, x0 = self._prepare_data(x, y, x0)
                
                output,_ = self.dcrnn_model(x, y, x0, batches_seen, True, self.z_mean_all, self.z_var_temp_all)
                
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            
            y_preds = y_preds[:,:2214] # remove the repeated data
            y_truths = y_truths[:,:2214]
            
            y_preds_scaled = np.exp(y_preds) - 1.
            y_truths_scaled = np.exp(y_truths) - 1.

            mae_metric, rmse_metric = self._test_loss(y_truths_scaled, y_preds_scaled)
            self._writer.add_scalar('{} loss'.format(dataset), mae_metric, batches_seen) #check

            return mae_metric, rmse_metric, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        self.num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(self.num_batches))

        batches_seen = self.num_batches * self._epoch_num

        # with autograd.detect_anomaly():
        saved_model = dict()
        saved_output = dict()
        saved_mae = None
        saved_rmse = None
        saved_epoch = None
        saved_z_total = None
        saved_val_mae = None

        for epoch_num in range(self._epoch_num, epochs):

            self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            mae_losses = []
            kld_losses = []
            z_mean_all_list = []
            z_var_temp_all_list = []

            start_time = time.time()


            for _, (x, y, x0) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y, x0 = self._prepare_data(x, y, x0)

                output, y_t, z_mean_all_sub, z_var_temp_all_sub, z_mean_context_sub, z_var_temp_context_sub = self.dcrnn_model(x, y, x0, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,gamma=lr_decay_ratio)
                mae_loss, kld_loss = self._compute_loss(y_t, output, z_mean_all_sub, z_var_temp_all_sub, z_mean_context_sub, z_var_temp_context_sub)
                loss = mae_loss + kld_loss

                self._logger.debug(loss.item())

                losses.append(loss.item())
                mae_losses.append(mae_loss.item())
                kld_losses.append(kld_loss.item())
                z_mean_all_list.append(z_mean_all_sub)
                z_var_temp_all_list.append(z_var_temp_all_sub)

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)
                optimizer.step()

            # calculate z_mean_all, and z_var_temp_all
            self.z_mean_all = torch.mean(torch.stack(z_mean_all_list,0),0)
            self.z_var_temp_all = torch.mean(torch.stack(z_var_temp_all_list,0),0)

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            val_loss, _, _ = self.evaluate(dataset='val', batches_seen=batches_seen, z_mean_all=self.z_mean_all, z_var_temp_all=self.z_var_temp_all)
            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, train_kld: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(mae_losses), np.mean(kld_losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_mae_loss, test_rmse_loss, test_outputs = self.evaluate(dataset='test', batches_seen=batches_seen, z_mean_all=self.z_mean_all, z_var_temp_all=self.z_var_temp_all)
                message = 'Epoch [{}/{}] ({}) test_mae: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           test_mae_loss, test_rmse_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)


            if val_loss < min_val_loss:
                wait = 0
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}.'
                    .format(min_val_loss, val_loss))
                min_val_loss = val_loss

                saved_model = self.dcrnn_model.state_dict()
                saved_val_mae = val_loss
                saved_mae = test_mae_loss
                saved_rmse = test_rmse_loss
                saved_outputs = test_outputs
                saved_epoch = epoch_num
                saved_z_total = torch.stack([self.z_mean_all, self.z_var_temp_all], dim=0)

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    model_file_name = self.save_model(saved_epoch, saved_z_total, saved_outputs, saved_model, saved_val_mae, saved_mae, saved_rmse)

                    self._logger.info(
                        'Final Val loss {:.4f}, Test MAE loss {:.4f}, Test RMSE loss {:.4f}, '
                        'saving to {}'.format(saved_val_mae, saved_mae, saved_rmse, model_file_name))
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

            if epoch_num == epochs-1:
                model_file_name = self.save_model(saved_epoch, saved_z_total, saved_outputs, saved_model, saved_val_mae, saved_mae, saved_rmse)

                self._logger.info(
                    'Final Val loss {:.4f}, Test MAE loss {:.4f}, Test RMSE loss {:.4f}, '
                    'saving to {}'.format(saved_val_mae, saved_mae, saved_rmse, model_file_name))


    def _prepare_data(self, x, y, x0):
        x, y, x0 = self._get_x_y(x, y, x0)
        x, y, x0 = self._get_x_y_in_correct_dims(x, y, x0)
        return x.to(device), y.to(device), x0.to(device)

    def _get_x_y(self, x, y, x0):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param x0: shape (batch_size, input_dim_startingpoint)
        :param y: shape (batch_size, seg_len, output_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 x0 shape (batch_size, input_dim_startingpoint)
                 y shape (seq_len, batch_size, output_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x0 = torch.from_numpy(x0).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        self._logger.debug("X0: {}".format(x0.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2)
        return x, y, x0

    def _get_x_y_in_correct_dims(self, x, y, x0):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param x0: shape (batch_size, input_dim_startingpoint)
        :param y: shape (horizon, batch_size, output_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 x0: shape (batch_size, input_dim_startingpoint)
                 y: shape (seq_len, batch_size, output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        return x, y, x0

    def _compute_loss(self, y_true, y_predicted, z_mean_all, z_var_temp_all, z_mean_context, z_var_temp_context):
        # mean = torch.from_numpy(self.standard_scaler_y.mean).float().to(device)
        # std = torch.from_numpy(self.standard_scaler_y.std).float().to(device)
        # y_true = (y_true * std) + mean
        # y_predicted = (y_predicted * std) + mean
        y_true = torch.exp(y_true) - 1.
        y_predicted = torch.exp(y_predicted) - 1.

        return mae_loss(y_predicted, y_true), kld_gaussian_loss(z_mean_all, z_var_temp_all, z_mean_context, z_var_temp_context)
        # return mae_loss(y_predicted, y_true), l2_mean_std_loss(z_mean_all, z_std_all, z_mean_context, z_std_context)
        #return mape_loss(y_predicted, y_true), l2_mean_std_loss(z_mean_all, z_std_all, z_mean_context, z_std_context)

    def _test_loss(self, y_true, y_predicted):

        return mae_metric(y_predicted, y_true), rmse_metric(y_predicted, y_true)




