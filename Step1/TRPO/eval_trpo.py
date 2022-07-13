import gym
from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import pandas as pd

def main():
    episodes = 50
    env = gym.make('CustomHopper-target-v0')
    #task = [3.53429174,3.92699082, 2.71433605, 10.0893801 ]
    #env.set_parameters(*task)
    print(env.get_parameters())
    from sb3_contrib import TRPO
    model_class = TRPO("MlpPolicy", env, gamma=0.99, verbose=1)
    model = model_class.load("/home/nicola/Desktop/MLDL/Step2/TRPO/results_old/TRPO_lr_0.001_gae_0.8_ntimesteps_2000000/2022-06-27_10-30-23/logs/trpo_model_2000000_steps.zip")
 # /home/nicola/Desktop/MLDL/Step2/TRPO/results_old/TRPO_lr_0.001_gae_0.8_ntimesteps_2000000/2022-06-27_10-30-23
    #evaluate_policy(model, env, n_eval_episodes=5, warn=False, render=True)
    obs = env.reset()
    mean_reward = []
    std_reward = 0

    for episode in range(episodes):
        done = False
        test_reward = 0
        step = 0

        while not done and step<500:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()
            #print("Reward x step: ", rewards)
            test_reward += rewards
            step = step + 1
            
        if dones:
            env.reset()
            mean_reward.append(test_reward)
            print(f"Episode: {episode} | Return: {test_reward}")
            
    print("Mean reward = " + str(np.mean(mean_reward)))
    print("Std = " + str(np.std(mean_reward)))            
    df = pd.DataFrame({'col':mean_reward})
    df.to_csv('test_trpo.csv', index=False) 

main()

   
