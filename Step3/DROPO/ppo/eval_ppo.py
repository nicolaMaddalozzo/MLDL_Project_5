import gym
from env.custom_hopper import *



def main():
    episodes = 10000
    env = gym.make('CustomHopper-source-v0')
    from stable_baselines3 import PPO
    model_class = PPO("MlpPolicy", env, gamma=0.9, verbose=1)
    model = model_class.load("ppo_hopper.mdl")
    #evaluate_policy(model, env, n_eval_episodes=5, warn=False, render=True)
    obs = env.reset()
    for episode in range(episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            env.reset()

main()

   
