import gym
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from env.custom_hopper import *
from stable_baselines3.common.logger import configure, read_csv
from stable_baselines3.common.callbacks import CheckpointCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-timesteps', default=200000, type=int, help='Number of training steps')
    parser.add_argument('--save-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--lr', default=0.001, type=float, help='hyperparameter: learning rate')
    parser.add_argument('--gae', default=0.95, type=float, help='hyperparameter: gae lambda factor')
    parser.add_argument('--results_folder', type=str, help='insert name of the folder where to put results')

    
    return parser.parse_args()


def main():
    env = gym.make('CustomHopper-source-v0')
    print(env.get_parameters())
    # Create folder for results
    if args.results_folder is not None:
        results_folder_path = args.results_folder_path
    else:
        results_folder_path = "results/TRPO_target_lr_" + str(args.lr) + "_gae_" + str(args.gae) + "_ntimesteps_" + str(args.n_timesteps)
        results_folder_path = os.path.join(results_folder_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    
    from sb3_contrib import TRPO
    
    if not os.path.exists(results_folder_path + "/logs"):
        os.makedirs(results_folder_path + "/logs") 
    if args.save_every is not None:
        checkpoint_callback = CheckpointCallback(save_freq=args.save_every, save_path=results_folder_path + "/logs/", name_prefix='trpo_model')
    else:
        checkpoint_callback = None
       

    model = TRPO("MlpPolicy", env, gae_lambda=args.gae, learning_rate=args.lr, gamma=0.99, verbose=1)
        
    tmp_path = results_folder_path + "/logs/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv"])

    # Set new logger
    model.set_logger(new_logger)
    
    model.learn(total_timesteps=args.n_timesteps, log_interval=1, callback=checkpoint_callback)
    if not os.path.exists(results_folder_path + "/model"):
        os.makedirs(results_folder_path + "/model") 
        model.save(results_folder_path + "/model/trpo_model.mdl")
    
    #print(model.logger.name_to_value['rollout/ep_rew_mean'])
    
    # Print and save plots
    if not os.path.exists(results_folder_path + "/graphs"):
        os.makedirs(results_folder_path + "/graphs")     
    
    csv = read_csv(results_folder_path + "/logs/progress.csv")
  
    x = csv.loc[:,"time/total_timesteps"]
    y = csv.loc[:,"rollout/ep_rew_mean"]
    plt.plot(x, y)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    fig = plt.gcf()
    fig.savefig(results_folder_path + "/graphs/cumulative_rewards.png", transparent=True, dpi=200)
    plt.show()


args = parse_args()
main()
