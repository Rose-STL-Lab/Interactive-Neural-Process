import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf
import random
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, x0s, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param x0s: (starting point)
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            x0_padding = np.repeat(x0s[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            x0s = np.concatenate([x0s, x0_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys, x0s = xs[permutation], ys[permutation], x0s[permutation]
        self.xs = xs
        self.ys = ys
        self.x0s = x0s

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x0_i = self.x0s[start_ind: end_ind, ...]
                yield (x_i, y_i, x0_i)
                self.current_ind += 1

        return _wrapper()


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters

def generate_new_trainset(selected_data, previous_data, dataset_dir, batch_size, test_batch_size=None, **kwargs):
    
    selected_x = np.concatenate(selected_data['x'],0)
    selected_y = np.concatenate(selected_data['y'],0)
    data1 = previous_data
    data1['x_train'] = np.concatenate([previous_data['x_train'], selected_x],0)
    data1['y_train'] = np.concatenate([previous_data['y_train'], selected_y[:,1:]],0)
    data1['x0_train'] = np.concatenate([previous_data['x0_train'], selected_y[:,0]],0)
    data1['train_loader'] = DataLoader(data1['x_train'], data1['y_train'], data1['x0_train'], batch_size, shuffle=True)

    return data1

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data_list = []
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        #normalize data
        data['y_' + category] = np.log(cat_data['y']+1.) 

    # load scenario index
    scenario_array = np.load(os.path.join(dataset_dir, 'train_scenario_array.npy'),allow_pickle=True)
    x_scenario_list = []
    y_scenario_list = []

    data['x_train'] = data['x_train'].reshape(9,1848,28, 58, 10)
    data['y_train'] = data['y_train'].reshape(9,1848, 29, 24)

    for i in range(len(scenario_array)):
        indices = np.array(scenario_array[i])
        scenario_x = data['x_train'][:,indices]
        scenario_y = data['y_train'][:,indices]

        x_scenario_list.append(scenario_x)
        y_scenario_list.append(scenario_y)

    # delet original train data
    del data['x_train']
    del data['y_train']

    # generate training data for initial case
    data['x_train'] = np.concatenate([x_scenario_list[26].reshape(-1,28, 58, 10),x_scenario_list[28].reshape(-1,28, 58, 10),
        x_scenario_list[30].reshape(-1,28, 58, 10)],0)
    data['y_train'] = np.concatenate([y_scenario_list[26].reshape(-1,29, 24),y_scenario_list[28].reshape(-1,29, 24),
        y_scenario_list[30].reshape(-1,29, 24)],0)

    # search data
    # search_data_list_x = list(x_scenario_list[:23] + x_scenario_list[24:])
    # search_data_list_y = list(y_scenario_list[:23] + y_scenario_list[24:])
    search_data_list_x = list(x_scenario_list[:26] + x_scenario_list[27:28] +
        x_scenario_list[29:30] + x_scenario_list[31:])
    search_data_list_y = list(y_scenario_list[:26] + y_scenario_list[27:28] +
        y_scenario_list[29:30] + y_scenario_list[31:])

    # flatten list
    search_data_x = [item for sublist in search_data_list_x for item in sublist]
    search_data_y = [item for sublist in search_data_list_y for item in sublist]


    # Data format (train data modified)
    data1 = {}
    for category in ['train', 'val', 'test']:
        data1['x_' + category] = data['x_' + category]
        data1['y_' + category] = data['y_' + category][:,1:]
        data1['x0_' + category] = data['y_' + category][:,0]


    data1['train_loader'] = DataLoader(data1['x_train'], data1['y_train'], data1['x0_train'], batch_size, shuffle=True)
    data1['val_loader'] = DataLoader(data1['x_val'], data1['y_val'], data1['x0_val'], test_batch_size, shuffle=False)
    data1['test_loader'] = DataLoader(data1['x_test'], data1['y_test'], data1['x0_test'], test_batch_size, shuffle=False)

    return data1, search_data_x, search_data_y


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
