"""
This file implement a a2c algorithm. We put Train.py and Agent.py toghether from the initial repository 
"""
import os
import torch
import gym
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from datetime import datetime
from env.custom_hopper import *
from logger import configure, Logger, HumanOutputFormat, KVWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="'source' or 'target' domain", required=True)
    parser.add_argument('--n-episodes', default=200000, type=int, help='Number of training episodes')
    parser.add_argument('--save-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--lr', default=0.001, type=float, help='hyperparameter: learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='hyperparameter: discount factor')
    parser.add_argument('--results_folder', default=None, type=str, help='insert name of the folder where to put results')
    
    return parser.parse_args()

def DiscRetBoos(gamma,next_states,value):
    
    next_values = value(next_states)
    
    return gamma*next_values



class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TODO 2.2.b: critic network for actor-critic algorithm


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TODO 2.2.b: forward in the critic network

        
        return normal_dist

class Value(torch.nn.Module):
    def __init__(self, state_space, var=1):
        super().__init__()
        self.state_space = state_space
        
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, var)

        
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                
    
    def forward(self, x):
        """
            Critic
        """
        # TODO 2.2.b: forward in the critic network

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic(x_critic)

        return value    

class Agent(object):
    def __init__(self, policy, value, device='cpu', lr=0.001, gamma=0.9):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.value = value.to(self.train_device)
        self.optimizer_act = torch.optim.Adam(policy.parameters(), lr=lr)
        self.optimizer_cri = torch.optim.Adam(value.parameters(), lr=lr)
        self.gamma = gamma
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def init_lists(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        
    def update_policy(self):     
        
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        #             - compute policy gradient loss function given actions and returns
       
        #value_states = self.value(states)
        value = self.value
        gamma = self.gamma
        d = 0.0
        A_hat = []
        n = len(rewards)
 
        for t in range(n):
            #A_hat.append(G_hat[t] - value_states[t].detach())
            #print(t)
            if(t==(n-1)):
                d = 1.0
            next_value = value(next_states[t])
            A_hat.append(rewards[t] - value(states[t]).detach() + (1.0 - d) * gamma * next_value) 
            #A_hat.append(rewards[t] + (1.0 - d) * gamma * value(next_states[t]) - value(states[t]) ) 
            
        loss_act = []
        #print(A_hat)
        A_hat = torch.stack(A_hat)#.squeeze(-1)
       
        
        
        for log_prob,At in zip(action_log_probs, A_hat.detach()):
            loss_act.append(-log_prob * At )
            
        loss_act = torch.sum(torch.stack(loss_act))
        #print(value_states.squeeze(-1).detach(), DiscountedRets.squeeze(-1))       
        
        #print(A_hat.grad_fn)
        
        loss_cri = torch.sum(-A_hat)#-F.mse_loss(value_states.squeeze(-1).detach(), DiscountedRets.squeeze(-1))       
        
        loss = loss_act + loss_cri
        #print(loss_act, loss_cri, loss)               
        self.optimizer_act.zero_grad()
        self.optimizer_cri.zero_grad()
        #print(loss)
        loss.backward()
        self.optimizer_act.step()
        self.optimizer_cri.step() 
        self.init_lists()
        
        return loss
        
        

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


        
def main():

    # Create folder for results
    if args.results_folder is not None:
        results_folder_path = args.results_folder
    else:
        results_folder_path = "results/a2c_" + args.env + "_lr" + str(args.lr) + "_gamma_" + str(args.gamma) + "_nepisodes_" + str(args.n_episodes)
        results_folder_path = os.path.join(results_folder_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    tmp_path = results_folder_path + "/logs/"
    # Set up logger
    logger = configure(tmp_path, ["csv"])
    logger_stdo = configure(tmp_path, ["stdout"])

    if args.env == "source":
        env = gym.make('CustomHopper-source-v0')
    elif args.env == "target":
        env = gym.make('CustomHopper-target-v0')
    else:
        print("No domain set, inser in the command line if you want to train on 'source' or 'target' domain")

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    #print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    
    policy = Policy(observation_space_dim, action_space_dim)
    value = Value(observation_space_dim,1)
    agent = Agent(policy,value, device=args.device, lr=args.lr, gamma=args.gamma)
    
    cumulative_rewards = []
    mean_rewards = []
    losses = []
    len_episodes = []
   
    for episode in range(args.n_episodes):
        
        #print(episode)
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state
        step=0
       
        while not done:  # Loop until the episode is over
            
            #action, action_probabilities = agent.get_action(state)
            
            x = torch.from_numpy(state).float().to('cpu')
         
            normal_dist = policy(x)
             
            evaluation = False 
            if evaluation:  # Return mean
                return normal_dist.mean, None

            else:   # Sample from the distribution
                action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            
            action_log_prob = normal_dist.log_prob(action)
            
            previous_state = state          

            state, reward, done, info = env.step(action.detach().cpu().numpy())
            
            agent.store_outcome(previous_state, state, action_log_prob, reward, done)
            step+=1
            train_reward += reward
           
             
        len_episodes.append(step)
        loss = agent.update_policy()
        
        losses.append(loss.detach().numpy())
        cumulative_rewards.append(train_reward)
        mean_rewards.append(np.mean(cumulative_rewards))
        
        
        # Logs
        logger.record("current/episode", episode)
        
        logger.record("current/ep_len", step)
        logger.record("current/ep_reward", train_reward)
        logger.record("train/loss", np.mean(losses))
        logger.record("train/ep_len_mean", np.mean(len_episodes))
        logger.record("train/ep_rew_mean", np.mean(cumulative_rewards))

        logger.dump(episode)
        
        #if (episode+1)%10 == 0:    
            #print(episode,step,train_reward)
            
            
        if (episode+1)%(args.save_every // 10) == 0:     
            logger_stdo.record("current/episode", episode)
            logger_stdo.record("current/ep_len", step)
            logger_stdo.record("current/ep_reward", train_reward)
            logger_stdo.record("train/loss", np.mean(losses))
            logger_stdo.record("train/ep_len_mean", np.mean(len_episodes))
            logger_stdo.record("train/ep_rew_mean", np.mean(cumulative_rewards))         
            logger_stdo.dump(episode)            
                  
        if (episode)%args.save_every == 0:
                       
            print('Training episode:', episode)
            print('Episode return:', train_reward)
            if not os.path.exists(results_folder_path + "/model"):
                os.makedirs(results_folder_path + "/model") 
            torch.save(agent.policy.state_dict(), results_folder_path + "/model/model_episode_" + str(episode) + ".mdl")
            
            
    # Print and save plots
    if not os.path.exists(results_folder_path + "/graphs"):
        os.makedirs(results_folder_path + "/graphs")     
    
    plt.plot(range(args.n_episodes), mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    fig = plt.gcf()
    fig.savefig(results_folder_path + "/graphs/cumulative_rewards.png", transparent=True, dpi=200)
    plt.show()
    
args = parse_args()
main()
