#!/usr/bin/python3
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

# This script reproduces the results for the real world datasets used in the paper

import importlib.util
spec = importlib.util.find_spec('silence_tensorflow')
if spec is not None:
    import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from helper import *
from models import DMGP_model, DFGP_model
from datetime import datetime

def run_dmgp_model(FLAGS_in):     
    print('\n#####   DMGP    #####')
    
    # Define arrays to store values of RMSE, NLPD, and training time for each split
    rmse_vec = np.zeros(FLAGS_in.num_splits)
    nlpd_vec = np.zeros(FLAGS_in.num_splits)
    train_time_vec = np.zeros(FLAGS_in.num_splits)
    
    for split_id in range(FLAGS_in.num_splits): 
        # for reproducibility    
        tf.random.set_seed(1) 
        np.random.seed(1) 
        
        # Load standardized dataset
        x_train, x_test, y_train, y_test = load_dataset_train_test_split(FLAGS_in.dataset, test_size = 0.1, split_id=split_id)

        print('\n*** Split: %d ***' %split_id)

        # Initialize model's parameters
        eps_sq_0, sigma_n_sq_0, sigma_f_sq_0, dir_weights_ = initialize_dmgp(x_train, y_train, FLAGS_in, split_id)
        
        # Define model
        model_dmgp = DMGP_model(data=(x_train, y_train), m=FLAGS_in.m_dmgp, d=FLAGS_in.d_dmgp, eps_sq=eps_sq_0, sigma_n_sq=sigma_n_sq_0,
                                                          sigma_f_sq=sigma_f_sq_0, dir_weights=dir_weights_)

        # Split parameters to those corresponding to kernel and DNN
        list_trainable_vars_kernel = tuple([var  for var in model_dmgp.trainable_variables if var.name == 'Variable:0'])
        list_trainable_vars_dnn = tuple([var  for var in model_dmgp.trainable_variables if var.name != 'Variable:0'])

        # Set minibatch size
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(buffer_size=y_train.size)
        batches_iter = iter(train_dataset.batch(FLAGS_in.batch_size))
        iters_per_epoch = y_train.size // FLAGS_in.batch_size
        
        # Define learning rates for kernel and DNN parameters separately
        lrate_kernel = 1e-6 if FLAGS.use_dnn_init else 1e-4
        lrate_dnn = 1e-6 if FLAGS.use_dnn_init else 1e-3
        
        # Define Adam optimizers for kernel and DNN parameters separately
        adam_opt_kernel = tf.optimizers.Adam(learning_rate=lrate_kernel)
        adam_opt_dnn = tf.optimizers.Adam(learning_rate=lrate_dnn)

        # Differentiation of negative log marginal likelihood with respect to kernel parameters
        @tf.function(autograph=False)
        def optimization_step_kernel():
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(list_trainable_vars_kernel)
                objective = model_dmgp.neg_log_marginal_likelihood()
                grads = tape.gradient(objective, list_trainable_vars_kernel)
            adam_opt_kernel.apply_gradients(zip(grads, list_trainable_vars_kernel))
            return objective
            
        # Differentiation of negative log marginal likelihood with respect to DNN weights
        @tf.function(autograph=False)
        def optimization_step_dnn():
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(list_trainable_vars_dnn)
                objective = model_dmgp.neg_log_marginal_likelihood(next(batches_iter))
                grads = tape.gradient(objective, list_trainable_vars_dnn)
            adam_opt_dnn.apply_gradients(zip(grads, list_trainable_vars_dnn))
            return objective
            
        print('Training...')
        start_time = datetime.now() 
        # Training
        for epoch in range(FLAGS_in.num_epochs):     
            for _ in range(iters_per_epoch):
                optimization_step_dnn()
            
            loss_value = optimization_step_kernel()
            if (epoch+1) % FLAGS.display_freq == 0 or epoch==0:
                print('Epoch: {}/{}   log-marginal: {:.3f}' .format(epoch+1, FLAGS_in.num_epochs, -loss_value.numpy()))
        train_time = datetime.now() - start_time
        print('Done')
        
        # Prediction
        f_mean, f_var = model_dmgp.predict_y(x_test)
        f_mean, f_var = f_mean.numpy(), f_var.numpy()
        
        # RMSE and NLPD computation
        rmse_test, nlpd_test = compute_rmse_nlpd(y_test, f_mean, f_var)
        rmse_vec[split_id], nlpd_vec[split_id], train_time_vec[split_id] = rmse_test, nlpd_test, train_time.total_seconds()

        print('\nDMGP  RMSE: %.3f    NLPD: %.3f' % (rmse_test, nlpd_test))  
        print('Training time  Full: {}  or {:.3f} seconds' .format(train_time, train_time.total_seconds()))
            
    print('\n\nDMGP-Averaged   RMSE: %.3f +/- %.3f   NLPD: %.3f +/- %.3f' % (rmse_vec.mean(), rmse_vec.std(), nlpd_vec.mean(), nlpd_vec.std()))  
    print('Average training time  %.3f +/- %.3f seconds' %(train_time_vec.mean(), train_time_vec.std()))
    
    
def run_dfgp_model(FLAGS_in):   
    print('\n\n#####   DFGP    #####') 
    
    # Define arrays to store values of RMSE, NLPD, and training time for each split
    rmse_vec = np.zeros(FLAGS_in.num_splits)
    nlpd_vec = np.zeros(FLAGS_in.num_splits)
    train_time_vec = np.zeros(FLAGS_in.num_splits)
    
    for split_id in range(FLAGS_in.num_splits):  
        # for reproducibility     
        tf.random.set_seed(1) 
        np.random.seed(1)    
        
        # Load standardized dataset
        x_train, x_test, y_train, y_test = load_dataset_train_test_split(FLAGS_in.dataset, test_size = 0.1, split_id=split_id)
        
        print('\n\n*** Split: %d ***' %split_id)
        
        lengthscales_0, sigma_n_sq_0, sigma_f_sq_0, dir_weights_ = initialize_dfgp(x_train, y_train, FLAGS_in, split_id)
        model_dfgp = DFGP_model(data=(x_train, y_train), m=FLAGS_in.m_dfgp, d=FLAGS_in.d_dfgp, lengthscales=lengthscales_0, sigma_n_sq=sigma_n_sq_0,
                                                          sigma_f_sq=sigma_f_sq_0, dir_weights=dir_weights_)

        list_trainable_vars_kernel = tuple([var  for var in model_dfgp.trainable_variables if var.name == 'Variable:0'])
        list_trainable_vars_dnn = tuple([var  for var in model_dfgp.trainable_variables if var.name != 'Variable:0'])

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(buffer_size=y_train.size)
        batches_iter = iter(train_dataset.batch(FLAGS_in.batch_size))
        iters_per_epoch = y_train.size // FLAGS_in.batch_size
        
        lrate_kernel = 1e-6 if FLAGS.use_dnn_init else 1e-4
        lrate_dnn = 1e-6 if FLAGS.use_dnn_init else 1e-3
        
        adam_opt_kernel = tf.optimizers.Adam(learning_rate=lrate_kernel)
        adam_opt_dnn = tf.optimizers.Adam(learning_rate=lrate_dnn)

        @tf.function(autograph=False)
        def optimization_step_kernel():
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(list_trainable_vars_kernel)
                objective = model_dfgp.neg_log_marginal_likelihood()
                grads = tape.gradient(objective, list_trainable_vars_kernel)
            adam_opt_kernel.apply_gradients(zip(grads, list_trainable_vars_kernel))
            return objective
            
        @tf.function(autograph=False)
        def optimization_step_dnn():
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(list_trainable_vars_dnn)
                objective = model_dfgp.neg_log_marginal_likelihood(next(batches_iter))
                grads = tape.gradient(objective, list_trainable_vars_dnn)
            adam_opt_dnn.apply_gradients(zip(grads, list_trainable_vars_dnn))
            return objective
            
        print('Training...')
        start_time = datetime.now() 
        for epoch in range(FLAGS_in.num_epochs):     
            for _ in range(iters_per_epoch):
                optimization_step_dnn()
            
            loss_value = optimization_step_kernel()
            if (epoch+1) % FLAGS.display_freq == 0 or epoch==0:
                print('Epoch: {}/{}   log-marginal: {:.3f}' .format(epoch+1, FLAGS_in.num_epochs, -loss_value.numpy()))
        train_time = datetime.now() - start_time
        print('Done')
        
        f_mean, f_var = model_dfgp.predict_y(x_test)
        f_mean, f_var = f_mean.numpy(), f_var.numpy()
        rmse_test, nlpd_test = compute_rmse_nlpd(y_test, f_mean, f_var)
        rmse_vec[split_id], nlpd_vec[split_id], train_time_vec[split_id] = rmse_test, nlpd_test, train_time.total_seconds()

        print('\nDFGP  RMSE: %.3f    NLPD: %.3f' % (rmse_test, nlpd_test))  
        print('Training time  Full: {}  or {:.3f} seconds' .format(train_time, train_time.total_seconds()))
            
    print('\n\nDFGP-Averaged   RMSE: %.3f +/- %.3f   NLPD: %.3f +/- %.3f' % (rmse_vec.mean(), rmse_vec.std(), nlpd_vec.mean(), nlpd_vec.std()))  
    print('Average training time  %.3f +/- %.3f seconds' %(train_time_vec.mean(), train_time_vec.std()))
    
    
# retrieve all flags
FLAGS = get_flags()

# Run the DMGP model
run_dmgp_model(FLAGS)

# Run the DFGP model
run_dfgp_model(FLAGS)
    