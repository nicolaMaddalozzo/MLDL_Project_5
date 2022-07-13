import gym
from env.custom_hopper import *



def main():
    episodes = 50
    env = gym.make('CustomHopper-target-v0')
    from stable_baselines3 import PPO
    model_class = PPO("MlpPolicy", env, gamma=0.9, verbose=1)
    
    model = model_class.load("/home/nicola/Desktop/MLDL/Step2/PPO/results_old/PPO_lr_0.001_gae_0.8_ntimesteps_2000000/2022-06-27_10-29-25/model/ppo_model.mdl")
#/home/nicola/Desktop/MLDL/Step2/PPO/results_old/PPO_lr_0.001_gae_0.8_ntimesteps_2000000/2022-06-27_10-29-25
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
        #    env.render()
            test_reward += rewards
            step = step + 1
        if dones:
            env.reset()
            mean_reward.append(test_reward)
            print(f"Episode: {episode} | Return: {test_reward}")
    print("Mean reward = " + str(np.mean(mean_reward)))
    print("Std = " + str(np.std(mean_reward)))
            
main()

   
