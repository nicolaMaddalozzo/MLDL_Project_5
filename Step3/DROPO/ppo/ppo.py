import gym
from env.custom_hopper import *



def main():
    env = gym.make('CustomHopper-source-v0')
    from stable_baselines3 import PPO
    
    model = PPO("MlpPolicy", env, gamma=0.99, verbose=1)
    model.learn(total_timesteps=2000000, log_interval=4)
    model.save("ppo_hopper.mdl")

main()    
  


 
