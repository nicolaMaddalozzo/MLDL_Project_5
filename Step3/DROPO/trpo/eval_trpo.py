import gym
from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def main():
    episodes = 50
    env = gym.make('CustomHopper-target-v0')
    from sb3_contrib import TRPO
    model_class = TRPO("MlpPolicy", env, gamma=0.99, verbose=1)
    model = model_class.load("results/TRPO_target_lr_0.0005_gae_0.8_ntimesteps_2000000/2022-06-28_23-19-51/model/trpo_model.mdl")
 # /home/nicola/Desktop/MLDL/Step2/TRPO/results/TRPO_target_lr_0.0005_gae_0.8_ntimesteps_2000000/2022-06-28_23-19-51
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


main()

   
