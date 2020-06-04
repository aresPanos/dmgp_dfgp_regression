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

# This script produces the results for the 1D-Data simulated datasets used in the paper

import importlib.util
spec = importlib.util.find_spec('silence_tensorflow')
if spec is not None:
    import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import gpflow
from tensorflow.keras import optimizers
from helper import *
from models import MGP_model, FGP_model
from datetime import datetime
tf.random.set_seed(1) # used for reproducibility

FLAGS = get_flags()
m_in = 34 # number of eigenfunctions used > 0
x_train, y_train, x_test, y_test = generate_1d() # generates data from f(x) = 1.5*sin(x/.6) + 0.5*cos(10*x) + x/8.
N_tr = y_train.size
dir_parent = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

# Exact GP inference
model_exactGP = gpflow.models.GPR(data=(x_train, y_train[:, None]), kernel=gpflow.kernels.SquaredExponential())

@tf.function(autograph=False)
def objective_closure():
    return -model_exactGP.log_marginal_likelihood()

opt = gpflow.optimizers.Scipy()
opt.minimize(objective_closure, model_exactGP.trainable_variables, options=dict(maxiter=500))
mean_f_gp, var_f_gp = model_exactGP.predict_y(x_test)
mean_f_gp, var_f_gp = mean_f_gp.numpy().squeeze(), var_f_gp.numpy().squeeze()
std_f_gp = np.sqrt(var_f_gp)

rmse_test_exactGP, nlpd_test_exactGP = compute_rmse_nlpd(y_test, mean_f_gp, var_f_gp)
marg_lkl_exactGP = model_exactGP.log_marginal_likelihood().numpy()
print('\nExact-GP RMSE: %.4f    NLPD: %.4f    Marginal: %.4f' % (rmse_test_exactGP, nlpd_test_exactGP, marg_lkl_exactGP))


# MGP inference
model_mgp = MGP_model(data=(x_train, y_train), m=m_in, eps_sq=5, sigma_n_sq=0.1)

@tf.function(autograph=False)
def objective_closure_mgp():
    return model_mgp.neg_log_marginal_likelihood()

opt_mgp = gpflow.optimizers.Scipy()
opt_mgp.minimize(objective_closure_mgp, model_mgp.trainable_variables, options=dict(maxiter=500))
marg_lkl_mgp = -model_mgp.neg_log_marginal_likelihood().numpy()
mean_f_mgp, var_f_mgp = model_mgp.predict_y(x_test)
mean_f_mgp, var_f_mgp = mean_f_mgp.numpy().squeeze(), var_f_mgp.numpy().squeeze()
std_f_mgp = np.sqrt(var_f_mgp)

rmse_test_mgp, nlpd_test_mgp = compute_rmse_nlpd(y_test, mean_f_mgp, var_f_mgp)
print('MGP RMSE: %.4f    NLPD: %.4f    Marginal: %.4f' % (rmse_test_mgp, nlpd_test_mgp, marg_lkl_mgp))


# FGP inference
model_fgp = FGP_model(data=(x_train, y_train), m=m_in, lengthscales=1, sigma_n_sq=5)

@tf.function(autograph=False)
def objective_closure_fgp():
    return model_fgp.neg_log_marginal_likelihood()

opt_mgp = gpflow.optimizers.Scipy()
opt_mgp.minimize(objective_closure_fgp, model_fgp.trainable_variables, options=dict(maxiter=500))
marg_lkl_fgp = -model_fgp.neg_log_marginal_likelihood().numpy()
mean_f_fgp, var_f_fgp = model_fgp.predict_y(x_test)
mean_f_fgp, var_f_fgp = mean_f_fgp.numpy().squeeze(), var_f_fgp.numpy().squeeze()
std_f_fgp = np.sqrt(var_f_fgp)

rmse_test_fgp, nlpd_test_fgp = compute_rmse_nlpd(y_test, mean_f_fgp, var_f_fgp)
print('FGP RMSE: %.4f    NLPD: %.4f    Marginal: %.4f' % (rmse_test_fgp, nlpd_test_fgp, marg_lkl_fgp))

fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(27,7))
ax[0].plot(x_train, y_train, 'kx', label='Training points', markersize=8, markeredgewidth=2)
ax[0].plot(x_test, y_test, 'r', label='Ground truth', linewidth=2)
ax[0].plot(x_test, mean_f_gp, color='y', dashes=[6, 2], linewidth=2, label='Exact GP')
ax[0].fill_between(x_test.squeeze(), mean_f_gp - 1.96*std_f_gp, mean_f_gp + 1.96*std_f_gp, facecolor='b', alpha=0.1)
ax[0].set_ylabel('Output $y$', fontsize=22)
ax[0].set_ylim([-3, 3])
ax[0].set_xlim([-2.5, 2.5])
ax[0].legend(loc='upper left', prop={'size': 16})

ax[1].plot(x_train, y_train, 'kx', label='Training points', markersize=8, markeredgewidth=2)
ax[1].plot(x_test, y_test, 'r', label='Ground truth', linewidth=2)
ax[1].plot(x_test, mean_f_mgp, color='y', dashes=[6, 2], linewidth=2, label='MGP - $r$=' + str(m_in))
ax[1].fill_between(x_test.squeeze(), mean_f_mgp - 1.96*std_f_mgp, mean_f_mgp + 1.96*std_f_mgp, facecolor='b', alpha=0.1)
ax[1].set_xlabel('Input $x$', fontsize=22)
ax[1].set_xlim([-2.5, 2.5])
ax[1].legend(loc='upper left', prop={'size': 16})

ax[2].plot(x_train, y_train, 'kx', label='Training points', markersize=8, markeredgewidth=2)
ax[2].plot(x_test, y_test, 'r', label='Ground truth', linewidth=2)
ax[2].plot(x_test.squeeze(), mean_f_fgp, color='y', dashes=[6, 2], linewidth=2, label='FGP - $r$=' + str(2*m_in))
ax[2].fill_between(x_test.squeeze(), mean_f_fgp - 1.96*std_f_fgp, mean_f_fgp + 1.96*std_f_fgp, facecolor='b', alpha=0.1)
ax[2].set_xlim([-2.5, 2.5])
ax[2].legend(loc='upper left', prop={'size': 16})

fig.tight_layout()
plt.savefig(dir_parent + '/plots/1d_example.png', format='png')

# Run MGP and FGP models with several values of m
if FLAGS.run_multiple_m:
    tf.random.set_seed(1) # used for reproducibility
    m_list = [2, 4, 8, 24, 30]

    for i in range(len(m_list)):
        model_mgp = MGP_model(data=(x_train, y_train), m=m_list[i], eps_sq=5, sigma_n_sq=0.1)

        @tf.function(autograph=False)
        def objective_closure_mgp():
            return model_mgp.neg_log_marginal_likelihood()

        opt_mgp = gpflow.optimizers.Scipy()
        opt_mgp.minimize(objective_closure_mgp, model_mgp.trainable_variables, options=dict(maxiter=500))
        marg_lkl_mgp = -model_mgp.neg_log_marginal_likelihood().numpy()
        mean_f_mgp, var_f_mgp = model_mgp.predict_y(x_test)
        mean_f_mgp, var_f_mgp = mean_f_mgp.numpy().squeeze(), var_f_mgp.numpy().squeeze()
        std_f_mgp = np.sqrt(var_f_mgp)

        print('\n\n*** # eigenfunctions/spectral densities: {} ***' .format(m_list[i]))
        rmse_test_mgp, nlpd_test_mgp = compute_rmse_nlpd(y_test, mean_f_mgp, var_f_mgp)
        print('MGP RMSE: %.4f    NLPD: %.4f    Marginal: %.4f' % (rmse_test_mgp, nlpd_test_mgp, marg_lkl_mgp))

        # FGP inference
        model_fgp = FGP_model(data=(x_train, y_train), m=m_list[i], lengthscales=1, sigma_n_sq=1)

        @tf.function(autograph=False)
        def objective_closure_fgp():
            return model_fgp.neg_log_marginal_likelihood()

        opt_mgp = gpflow.optimizers.Scipy()
        opt_mgp.minimize(objective_closure_fgp, model_fgp.trainable_variables, options=dict(maxiter=500))
        marg_lkl_fgp = -model_fgp.neg_log_marginal_likelihood().numpy()
        mean_f_fgp, var_f_fgp = model_fgp.predict_y(x_test)
        mean_f_fgp, var_f_fgp = mean_f_fgp.numpy().squeeze(), var_f_fgp.numpy().squeeze()
        std_f_fgp = np.sqrt(var_f_fgp)

        rmse_test_fgp, nlpd_test_fgp = compute_rmse_nlpd(y_test, mean_f_fgp, var_f_fgp)
        print('FGP RMSE: %.4f    NLPD: %.4f    Marginal: %.4f' % (rmse_test_fgp, nlpd_test_fgp, marg_lkl_fgp))

        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(17,7))
        ax[0].plot(x_train, y_train, 'kx', label='Training points', markersize=8, markeredgewidth=2)
        ax[0].plot(x_test, y_test, 'r', label='Ground truth', linewidth=2)
        ax[0].plot(x_test, mean_f_mgp, color='y', dashes=[6, 2], linewidth=2, label='MGP - $r$=' + str(m_list[i]))
        ax[0].fill_between(x_test.squeeze(), mean_f_mgp - 1.96*std_f_mgp, mean_f_mgp + 1.96*std_f_mgp, facecolor='b', alpha=0.1)
        ax[0].set_ylabel('Output $y$', fontsize=22)
        ax[0].set_ylim([-3, 3])
        ax[0].set_xlabel('Input $x$', fontsize=22)
        ax[0].set_xlim([-2.5, 2.5])
        ax[0].legend(loc='upper left', prop={'size': 16})

        ax[1].plot(x_train, y_train, 'kx', label='Training points', markersize=8, markeredgewidth=2)
        ax[1].plot(x_test, y_test, 'r', label='Ground truth', linewidth=2)
        ax[1].plot(x_test.squeeze(), mean_f_fgp, color='y', dashes=[6, 2], linewidth=2, label='FGP - $r$=' + str(2*m_list[i]))
        ax[1].fill_between(x_test.squeeze(), mean_f_fgp - 1.96*std_f_fgp, mean_f_fgp + 1.96*std_f_fgp, facecolor='b', alpha=0.1)
        ax[1].set_xlim([-2.5, 2.5])
        ax[1].set_xlabel('Input $x$', fontsize=22)
        ax[1].legend(loc='upper left', prop={'size': 16})
        plt.savefig(dir_parent + '/plots/1d_example_fgp_mgp_r=' + str(m_list[i]) + '.png', format='png')

plt.show()
