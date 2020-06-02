# Copyright 2020 Aristeidis Panos

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
spec = importlib.util.find_spec('silence_tensorflow')
if spec is not None:
    import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import os 
import gpflow
from models import MGP_model, FGP_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fun_1d(x_in):
    return 1.5*np.sin(x_in/.6) + 0.5*np.cos(x_in/.1) + x_in/8.
    
    
def generate_1d(N_tr: int = 25, N_test: int = 2000, random_gen: int = 2 ):
    np.random.seed(random_gen)
    x_train = np.random.randn(N_tr)
    x_train.sort()
    y_train = fun_1d(x_train) + 0.01*np.random.randn(N_tr)

    x_test = np.linspace(-2.5, 2.5, N_test)
    y_test = fun_1d(x_test)
    
    return x_train[:, None], y_train, x_test[:, None], y_test
    
    
def load_dataset_train_test_split(dataset_name, test_size = 0.1, split_id: int = 1):
    if dataset_name not in ['elevators', 'protein', 'sarcos', '3droad']:
        raise NameError('The dataset with name ' + dataset_name + ' is not valid. Available dataset names: elevators, protein, sarcos, 3droad.')

    dir_parent = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    data_all = np.load(dir_parent + '/data/' + dataset_name + '_all.npy', allow_pickle=True)
    x_all, y_all = data_all[()]['x_all'], data_all[()]['y_all']
    
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size, random_state=split_id)
    if split_id == 0:
        print('*** Dataset: {} ****' .format(dataset_name))
        print('N_train: {}   N_test: {}   D: {} \n' .format(y_train.size, y_test.size, x_train.shape[1]))
           
    if dataset_name != 'sarcos':
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    y_train_mean, y_train_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_train_mean)/y_train_std
    y_test = (y_test - y_train_mean)/y_train_std
        
    return x_train, x_test, y_train, y_test 
 
    
def compute_rmse_nlpd(y_true, y_pred, var_test):
    mse_term = np.square(y_true - y_pred)
    nlpd = 0.5*(np.log(2.*np.pi*var_test) + mse_term/var_test)
    rmse = np.sqrt(mse_term.mean())

    return rmse, nlpd.mean()
    
    
# Get flags from the command line
def get_flags():
    flags = tf.compat.v1.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 1000, 'Batch size')
    flags.DEFINE_integer('num_epochs', 100, 'Number of epochs used to train DMGP/DFGP model')
    flags.DEFINE_integer('display_freq', 10, 'Display loss function value every FLAGS.display_freq epochs')
    flags.DEFINE_integer('num_splits', 1, 'Number of random data splits used - number of experiments run for a model')
    flags.DEFINE_integer('m_dmgp', 15, 'Number of eigenfunctions/eigenvalues (per dimension) used for DMGP')
    flags.DEFINE_integer('d_dmgp', 1, 'Number of output dimensions for DMGP\'s DNN')
    flags.DEFINE_integer('m_dfgp', 20, 'Number of spectral frequencies used for DFGP')
    flags.DEFINE_integer('d_dfgp', 4, 'Number of output dimensions for DFGP\'s DNN')
    flags.DEFINE_string('dataset', 'elevators', 'Dataset name')
    flags.DEFINE_bool('run_multiple_m', True, 'Running multiple experiments over the synthetic dataset usinge several values of m')
    flags.DEFINE_bool('use_dnn_init', True, 'Use the initialization strategy described in the paper for both DMGP and DFGP')

    return FLAGS
       
   
def initialize_dmgp(x_train, y_train, FLAGS, split_id: int = 0):
    print('Initialization...')
    if FLAGS.use_dnn_init:
        dir = os.path.dirname(os.path.realpath('.'))
        dir_weights = dir + '/models/dnn_dmgp_weights_' + FLAGS.dataset + '_split=' + str(split_id+1)
        y_train_augm = y_train if FLAGS.d_dmgp == 1 else np.repeat(y_train[:, None], FLAGS.d_dmgp, axis=1)

        def dnn_dmgp():
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(512, activation='tanh', input_dim=x_train.shape[1]))
            model.add(tf.keras.layers.Dense(256, activation='tanh'))
            model.add(tf.keras.layers.Dense(64, activation='tanh'))
            model.add(tf.keras.layers.Dense(FLAGS.d_dmgp))
            return model
        
        adam_dnn = tf.optimizers.Adam(learning_rate=1e-3)
        dnn_model = dnn_dmgp()       
        dnn_model.compile(optimizer=adam_dnn, loss='mean_squared_error')
        dnn_model.fit(x_train, y_train_augm, epochs=100, batch_size=1000, verbose=False)

        tf.keras.backend.set_floatx('float64')
        ws = dnn_model.get_weights()
        wsp = [w.astype(tf.keras.backend.floatx()) for w in ws]
        dnn_model_double = dnn_dmgp()
        dnn_model_double.set_weights(wsp)
        dnn_model_double.save_weights(dir_weights)

        x_tr_nn = dnn_model.predict(x_train, workers=16, use_multiprocessing=True).astype(np.float64)
        mgp_model = MGP_model(data=(x_tr_nn, y_train), m=FLAGS.m_dmgp)

        @tf.function(autograph=False)
        def objective_closure():
            return mgp_model.neg_log_marginal_likelihood()
        opt = gpflow.optimizers.Scipy()

        opt.minimize(objective_closure, mgp_model.trainable_variables, options=dict(maxiter=100))
        
        print('Done')
        return mgp_model.eps_sq.value().numpy(), mgp_model.sigma_n_sq.value().numpy(), mgp_model.sigma_f_sq.value().numpy(), dir_weights
    
    else:
        eps_sq_0, sigma_n_sq_0, sigma_f_sq_0 = 0.5, 0.1, 1      
        print('Done')
        return eps_sq_0, sigma_n_sq_0, sigma_f_sq_0, None
        

def initialize_dfgp(x_train, y_train, FLAGS, split_id: int = 0):
    print('Initialization...')
    if FLAGS.use_dnn_init:
        dir = os.path.dirname(os.path.realpath('.'))
        dir_weights = dir + '/models/dnn_dfgp_weights_' + FLAGS.dataset + '_split=' + str(split_id+1)
        y_train_augm = y_train if FLAGS.d_dfgp == 1 else np.repeat(y_train[:, None], FLAGS.d_dfgp, axis=1)
        
        def dnn_dfgp():
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(512, activation='tanh', input_dim=x_train.shape[1]))
            model.add(tf.keras.layers.Dense(256, activation='tanh'))
            model.add(tf.keras.layers.Dense(64, activation='tanh'))
            model.add(tf.keras.layers.Dense(FLAGS.d_dfgp))
            return model
        
        adam_dnn = tf.optimizers.Adam(learning_rate=1e-3)
        dnn_model = dnn_dfgp()       
        dnn_model.compile(optimizer=adam_dnn, loss='mean_squared_error')
        dnn_model.fit(x_train, y_train_augm, epochs=100, batch_size=1000, verbose=False)
        tf.keras.backend.set_floatx('float64')
        ws = dnn_model.get_weights()
        wsp = [w.astype(tf.keras.backend.floatx()) for w in ws]
        dnn_model_double = dnn_dfgp()
        dnn_model_double.set_weights(wsp)
        dnn_model_double.save_weights(dir_weights)
        
        x_tr_nn = dnn_model.predict(x_train, workers=16, use_multiprocessing=True).astype(np.float64)
        fgp_model = FGP_model(data=(x_tr_nn, y_train), m=FLAGS.m_dfgp)

        @tf.function(autograph=False)
        def objective_closure():
            return fgp_model.neg_log_marginal_likelihood()
        opt = gpflow.optimizers.Scipy()

        opt.minimize(objective_closure, fgp_model.trainable_variables, options=dict(maxiter=100))
        print('Done')
        return fgp_model.lengthscales.value().numpy(), fgp_model.sigma_n_sq.value().numpy(), fgp_model.sigma_f_sq.value().numpy(), dir_weights
    
    else:
        lengthscales_0, sigma_n_sq_0, sigma_f_sq_0 = 1, 0.1, 1
        print('Done')        
        return lengthscales_0, sigma_n_sq_0, sigma_f_sq_0, None