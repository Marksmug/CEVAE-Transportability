# Transfer Learning in Latent Contextual Bandits with Covariate Shift Through Causal Transportability
This repo is the code for the experiment in the paper [Transfer Learning in Latent Contextual Bandits with Covariate Shift Through Causal Transportability](https://arxiv.org/abs/2502.20153), which will be presented in the conference [CLeaR 2025](https://www.cclear.cc/2025).

The implementation of CEVAE is based on the paper [A Critical Look at the Consistency of Causal Estimation with Deep Latent Variable Models](https://github.com/severi-rissanen/critical_look_causal_dlvms).  

## Instruction
The experiment with proxy IHDP dataset and proxy MNIST dataset can be done by running 
- `.VAE_transfer/IHDP_experiment.ipynb`
- `.VAE_transfer/MNIST_experiment.ipynb`

 The two notebooks reproduce the experiment results in the paper following six steps:
1. Loading dataset (optional)
2. Training data-generating models (optional)
3. Generating data
4. Pre-training VAEs with source domain data
5. Evaluating the performance with target domain data 
6. Plotting results

The first two steps are optional since we have included pretrained data-generating models in `.VAE_transfer/datageneratormodels/`. To run the first steps, you need to download corresponding datasets to `.VAE_transfer/Data/IHDP` or `.VAE_transfer/Data/MNIST`, which can be downloaded from 
- [IHDP](https://www.fredjo.com/)
- [MNIST](http://yann.lecun.com/exdb/mnist/)

The MNIST data-generating models trainning is done in a separate notebook `.VAE_transfer/GAN_model_tranining.ipynb`. 

The third step generating data for the experiment using data-generating models and divide the source domain and target domain in three ways:
1. Separate by a binary covariate in the proxy
2. Separate by truncating the data on the latent context
3. Separate by sampling two distinguish Gaussians on the laten context

The last two ways are used in the paper.