# MAPPO with recurrent policies

This repository implements MAPPO, a multi-agent variant of PPO. The implementation in this repositorory is heavily based on: https://github.com/marlbenchmark/on-policy. The objective of this project is to compare a RIMs policy with an LSTM policy in a broad array of environments.

Integration with PettingZoo (https://pettingzoo.farama.org/) is supported.


## 1. Usage
## 1. Usage

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code. 

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 
* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 
* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 

More specifically:
* In onpolicy-algorithms-r_mappo-algorithm, there are two scripts:
- r_actor_critic contains the actor and the critic, with RIMs and LSTM
- rMAPPOPolicy is a wrapper for the actor and the critic,  which allows  to retrieve actions and values as well as to evaluate the actions

* In onpolicy-algorithms-r_mappo:
- the script r_mappo is the trainer class, which can calculate the value function loss, and crucially perform a training update using minibatch GD. It takes in the advantages from the buffer, it generates samples using a generator (either recurrent or feedforward) and performs a PPO update on each of those samples.

* In onpolicy-algorithms-utils:
- the script RIM contains the implementation of RIMs

* In onpolicy-runner-separated, there are two scripts:

- base_runner, which is the main runner. It takes in the wrapper for the actor  and the critic as well as the trainer class. It's where the handling of the multiple agents happens. It computes returns and trains the agents.
- mpe_runner, for each step of each episode, it executes the step function, computes the attention bonuses and inserts data into the buffer. Then, it computes returns and updates the network



## 2. Installation

Example installation. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
virtualenv marl
source marl/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
pip install box2d box2d-kengz
pip install wandb
pip install tensorboardX
pip install imageio


We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 3.Train

Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

```

