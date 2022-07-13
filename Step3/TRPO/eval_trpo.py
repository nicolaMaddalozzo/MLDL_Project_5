import gym
from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import datetime
import os
import pandas as pd

def main():
    episodes = 50
    env = gym.make('CustomHopper-source-v0')
    #env.set_parameters([3.53429174, 3.92699082, 2.71433605, 10.0893801 ])
    
    from sb3_contrib import TRPO
    model_class = TRPO("MlpPolicy", env, gamma=0.99, verbose=1)
    model = model_class.load("/home/nicola/Desktop/MLDL/Step4/TRPO/results_old/lambda_15_genSource/2022-07-09_12-37-38/model/trpo_model.mdl")
    #/home/nicola/Desktop/MLDL/Step4/TRPO/results_old/lambda_15_genSource/2022-07-09_12-37-38
    obs = env.reset()
    mean_rewards = []
    std_reward = 0
    for episode in range(episodes):
        done = False
        test_reward = 0
        step = 0
        while not done and step<500:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()
            test_reward += rewards
            step = step + 1
        
        if dones:
            env.reset()
            mean_rewards.append(test_reward)
            print(f"Episode: {episode} | Return: {test_reward}")
            
    print("Mean reward = " + str(np.mean(mean_rewards)))
    print("Std = " + str(np.std(mean_rewards)))  
    df = pd.DataFrame({'col':mean_rewards})
    df.to_csv('conf3_ADR_source_rob.csv', index=False)   
    """
    # Print and save plots
    if not os.path.exists(os.getcwd() + "/graphs"):
        os.makedirs(os.getcwd() + "/graphs")     
    
    plt.plot(range(episodes), mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    fig = plt.gcf()
    results_folder_path = os.path.join(os.getcwd() + "/graphs", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    fig.savefig(results_folder_path + "/cumulative_rewards_test.png", transparent=True, dpi=200)
    plt.show()          
    """
main()

   
