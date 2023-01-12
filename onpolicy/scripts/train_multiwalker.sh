#!/bin/sh
env="SISL-multiwalker"
scenario="multiwalker" 
num_agents=3
algo="rmappo"
exp="check"
seed_max=1
num_units=10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_multiwalker.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_units ${num_units} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 100 --num_env_steps 20000000 --ppo_epoch 4 --use_ReLU --gain 0.01 --lr 1e-5 --critic_lr 1e-5 --wandb_name "bolottasamuele" --user_name "bolottasamuele"
done
