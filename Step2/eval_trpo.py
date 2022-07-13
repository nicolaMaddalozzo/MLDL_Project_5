import gym
import pandas as pd
from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3.common.logger import configure, read_csv

def main():
# Create folder for results
    
        
    episodes = 50
    env = gym.make('CustomHopper-target-v0')
    from sb3_contrib import TRPO
    model_class = TRPO("MlpPolicy", env, gamma=0.99, verbose=1)
    
    
    
    
    model = model_class.load("/home/nicola/Desktop/MLDL/Step3/TRPO/results_old/TRPO_target_lr_0.001_gae_0.8_ntimesteps_2000000_CONF3/2022-07-05_15-54-46/model/trpo_model.mdl")
#/home/nicola/Desktop/MLDL/Step3/TRPO/results_old/TRPO_target_lr_0.001_gae_0.8_ntimesteps_2000000_CONF3/2022-07-05_15-54-46
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
    df.to_csv('conaxsad.csv', index=False)       


main()
   
