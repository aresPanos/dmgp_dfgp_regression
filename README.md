# Deep Mercer Gaussian Process (DMGP) and Deep Fourier Gaussian Process (DFGP) Regression #

<img src="plots/1d_example.png" height="300">

We provide the code used in our paper [Faster Gaussian Processes via Deep Embeddings](https://arxiv.org/abs/2004.01584) to reproduce results. The Deep Mercer GP,  Deep Fourier GP, and their corresspnding shallow counterparts models are implemented. 

## Requirements ##
TensorFlow - version 2.1.0  
TensorFlow Probability - version 0.9.0  
GPflow - version 2.0.0 or newer  
silence-tensorflow - version 1.1.1 (optional)

## Flags ##
* batch_size: Batch size (integer - default=1000)
* num_epochs: Display loss function value every FLAGS.display_freq epochs (integer - default=100)
* num_splits: Number of random data splits used - number of experiments run for a model (integer - default=1)
* m_dmgp: Number of eigenfunctions/eigenvalues (per dimension) used for DMGP (integer - default=15)
* d_dmgp: Number of output dimensions for DMGP's DNN (integer - default=1)
* m_dfgp: Number of spectral frequencies used for DFGP (integer - default=20)
* d_dfgp: Number of output dimensions for DFGP's DNN (integer - default=4)
* dataset: Dataset name (string - available names=[elevators, protein, sarcos, 3droad] - default=elevators)
* run_multiple_m: Running multiple experiments over the synthetic dataset usinge several values of m (boolean - default=True)
* use_dnn_init: Use the initialization strategy described in the paper for both DMGP and DFGP (boolean - default=True)

## Source code ##

The following files can be found in the **src** directory :  

- *models.py*: implementation of all the models used (DMGP/MGP  and DFGP/FGP)
- *helper.py*: various utility functions
- *hermite_coeff.npy*: a numpy array containing the Hermite polynomial coefficients needed for the DMGP model
- *main_realData.py*: code for replicating the results of DMGP and DFGP over the real-world datasets
- *main_toyData.py*: code for replicating the results over the 1D-toy dataset

The **models** folder contains saved models needed for the initialization of the DMGP/DFGP while **plots** folder contains plots generated by running the python file *main_toyData.py*.

## Examples ##
Some representive examples to run experiments using DMGP/DFGP model.

```
# Train MGP and FGP model on the toy dataset for fixed number of eigenfunctions and spectral frequencies, respectively.
# A pdf file called *1d_example.png* is generared in plots folder.

python src/main_toyData.py --run_multiple_m=False

```

```
# Learn DMGP and DFGP models over the Protein dataset and repeat experiments 5 times.
# Set the number of epochs equal to 100 and use mini-batches of size 1000. 
# Print the values of the log-marginal likelihood every 10 epochs.
# For DMGP use 15 eigenfunctions with 1-D embedding while for DFGP use 20 spectral frequencies with a 4-D embedding.
# For both models, initialize their DNN's weights with pretrained DNNs.

python src/main_realData.py --dataset=protein --batch_size=1000 --display_freq=10 --num_splits=5 --m_dmgp=15 --d_dmgp=1  --m_dfgp=20 --d_dmgp=4 --use_dnn_init=True

```


