import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.autograd import Variable


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


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
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

        #
        # TODO 2.2.a:
        #             - compute discounted returns
        #             - compute policy gradient loss function given actions and returns
        #             - compute gradients and step the optimizer
        #
        ####             - compute discounted returns
        #rewards = np.array([1,2,3,4,5,6,7,8,9,10])
        """
for t in range(len(rewards)):
    G = 0.0
    esp = 1
    for i in range(len(rewards[t:])):
        if i==0:
            G+=rewards[i]
        else :
            G = G + rewards[i]*pow(gamma,esp)   
            esp+=1
    DiscountedReturns = np.append(DiscountedReturns, G) 
        """
        DiscountedReturns = np.array([],dtype=float)
        for t in range(len(rewards)):
            G = 0.0
            esp = 1
            for i in range(len(rewards[t:])):
                if i==0:
                    G+=rewards[i]
                else :
                    G = G + rewards[i]*pow(self.gamma,esp)    
                    esp+=1
            DiscountedReturns = np.append(DiscountedReturns, G) 
        #print(len(rewards))
        #print(len(DiscountedReturns))
        #print(len(states))
        #print((action_log_probs))
        
        #"""    
        #   for k,r in enumerate(rewards[t:]):
        #        G += (self.gamma**k)*r
        #    DiscountedReturns= np.append(DiscountedReturns,G) #array of returns 
        #print((DiscountedReturns))   
        #"""
        ####
        
        ####            - compute policy gradient loss function given actions and returns
        
        #loss=torch.zeros(len(states), )        
        loss1=np.array([])                   # /*change States to states (states is above (1#))*/
      
        for State, action_log_prob, G in zip(states, action_log_probs, DiscountedReturns):
            #probs = self.policy()                                           #
            
            #/*I comments these 2 lines because I think that dist and log_prob are already implemented above (2#) */ 
            dist = self.policy.forward(State)         # In these two steps we obtain a distribution from the NN
            log_prob = dist.log_prob(action_log_prob)           

            loss = - log_prob*G #Loss! we put a negative sign because the theorem gradient descent want to find a minimum (so we put -)
            #print(type(loss))
            loss1 = np.append(loss1,loss.detach().numpy())
            a = torch.tensor(loss1, requires_grad=True)
            loss2 = torch.mean(a)
            loss2 = Variable(loss2.data, requires_grad=True)
        ###CERCARE DI CAPIRE COME SI SOMMANO/METTONO INSIEME TRA DI LORO LE LOSS DEI DIVERSI STEP DEL CICLO FOR
            
        #####
            
        #####         - compute gradients and step the optimizer
            self.optimizer.zero_grad()
            loss2.backward()
            self.optimizer.step() #update the parameters
            #index+=index
            #if(index%1000000==0):
             #   print(1000000)
                
        #

        #
        # TODO 2.2.b:
        #             - compute boostrapped discounted return estimates
        #             - compute advantage terms
        #             - compute actor loss and critic loss
        #             - compute gradients and step the optimizer
        #

        return        

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        #self.done.append(done)
        #self.states=(torch.from_numpy(state).float())
        #self.next_states=(torch.from_numpy(next_state).float())
        #self.action_log_probs=(action_log_prob)
        #self.rewards=(torch.Tensor([reward]))
        self.done.append(done)


