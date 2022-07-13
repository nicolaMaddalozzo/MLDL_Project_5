from prettytable import PrettyTable
import torch
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.autograd import Variable
import mujoco_py
from env.custom_hopper import *
import nevergrad as ng
import cma
from torch_truncnorm.TruncatedNormal import TruncatedNormal
from scipy.stats import multivariate_normal
from trpo.sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy



def get_data_from_sim(env=None, model=None, dim=500, render=False, algo=None):

    real_act = []
    real_obs =  []
    real_term_states =  []
    
    steps = dim
    
    # Initialize environment and model used for data generation, if already available
    if env is None:
    	env = gym.make('CustomHopper-target-v0')
    if algo is 'PPO':
        model_class = PPO("MlpPolicy", env, gamma=0.9, verbose=1)
        if model is None:
    	    model = model_class.load("TRPO_PPO/ppo/ppo_model.mdl")
    elif algo is 'TRPO' or algo is None:
        model_class = TRPO("MlpPolicy", env, gamma=0.99, verbose=1)
        if model is None:
    	    model = model_class.load("trpo/trpo_model.mdl")
        else:
            model = model_class.load(model)
    	         

    # Target to learn
    target_task = env.get_parameters()
    print('Initial dynamics:', target_task)
    
    # Generate data for future optimization phase
    obs = env.reset()

    for step in range(steps):

        real_obs.append(obs)
        action, _states = model.predict(obs)
        real_act.append(action)
        obs, rewards, dones, info = env.step(action)
        real_term_states.append(dones)
        
        if render:
            env.render()
        if dones:
            env.reset()  
            
    return np.array(real_act), np.array(real_obs), np.array(real_term_states), target_task

def determ(cov_matr_s):
    det = np.linalg.det(np.array(cov_matr_s))
    if(det<=0):
        det = 0.01
    return det            
                 
def likelihood_f(psi_mean_vector, real_act, real_obs, real_term_states, env):
    done = False
    #tau=0.0001
    tot_sets = 50
    likelihood = 0

    epsilons_set = np.zeros((tot_sets,4))
    lambda_hyp = 15
    T = len(real_obs)
    list_states_tlambd =[]
    list_real_states_tlambd = [] 

    eps = 0.1
    index = 0
        
    epsilon = np.zeros((11, 11), float)
    np.fill_diagonal(epsilon, eps)

    treshold = T-lambda_hyp
    mean_s = np.array([10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],dtype=float)
    s_tPl =  np.array([11,11,11,11,11,11,11,11,11,11,11.0],dtype=float)
    
    # Generate set of environment parameters (epsilons) for optimization phase 
    for K in range(tot_sets):
    	for kk in range(len(epsilons_set[K])):
    	    
    	    # Epsilons are sampled from a truncated normal described by the distribution parameters to be optimized
            epsilons_set[K, kk] = TruncatedNormal(psi_mean_vector[kk], psi_mean_vector[kk+4], 0, 10).rsample()

    # Run trajectories of lambda steps, until T-lambda element
    for t in range(T-lambda_hyp):
        if(real_term_states[t]==False and done==False): #If the state t is not terminal
            
            state_t = real_obs[t]
            target_state = real_obs[t+lambda_hyp]
            
            # Run a trajectory for each tuple of epsilons in the set of epsilons
            env.reset()
            
            for K in range(tot_sets):
                 
                              
                env.set_mujoco_state(torch.tensor(state_t))
                env.sim.forward()
                
                # Set environmental parameters (tuple of epsilons)
                env.set_parameters(epsilons_set[K])                
               
                actions = real_act[t:(t+lambda_hyp)] #I don't put -1 because python take from action t from action t+lambda_hyp-1 with this line
                
                # Run the single trajectory from t to t+lambda              
                for i in range(lambda_hyp):

                    state_tlambd, _, done, info = env.step(torch.tensor(actions[i]).detach().cpu().numpy())
                
                list_states_tlambd.append(state_tlambd)
                
            # Calculate the mean and covariance matrix of the target states reached using the different epsilons
            mean_s = np.mean(list_states_tlambd, axis=0)
            cov_matr_s = np.cov(np.array(list_states_tlambd).T) + epsilon
            
            mask = (cov_matr_s<=0) #mask
            cov_matr_s[mask] = 0  #I set to 0 the correlations that are < 0 
 
            # Extraction of estimated states from a multivariate normal with mean = mean_s and covariance matrix = cov_matr_s
            s_tPl = np.random.multivariate_normal((mean_s), (cov_matr_s)) 
            list_real_states_tlambd.append(s_tPl)
            
            # Likelihood function calculation 
            multi_normal = multivariate_normal(mean=mean_s, cov=cov_matr_s)
            logdensity = multi_normal.logpdf(target_state)
            
            likelihood = likelihood + logdensity
    
    return (-1)*likelihood # We need to maximize, but the algorithm performs minimization, so we return -1*likelihood


print("Select algorithm policy used to generate data: ")
select_algo = input("c --> TRPO \nd --> PPO\n")
while select_algo not in ['c','d']:
    select_algo = input("ERROR! Please provide the answer writing 'c', 'd': \n")
if select_algo is 'c':
    selected_algo = 'TRPO'
elif select_algo is 'd':
    selected_algo = 'PPO'
    
yn = input("Do you want to indicate a specific model to generate data? (y/n) ")
while yn not in ['y','n']:
    yn = input("ERROR! Please provide the answer writing 'y' or 'n': ")
if yn is 'y':
    model_address = input("Write the model complete path (c://..//..//model.mdl): ") 
elif yn is 'n':
    model_address = None

st = input("Do you want to run on source (s) or target (t) domain? (s/t) ")
while st not in ['s','t']:
    st = input("ERROR! Please provide the answer writing 's' or 't': ")
if st is 's':
    domain = 'CustomHopper-source-v0'
elif st is 't':
    domain = 'CustomHopper-target-v0'
    
dim = input("How many data points do you want? Please write the number, minimum 100: \n")
while not dim.isdigit() or int(dim) < 100:
    dim = input("ERROR! Please write the number, minimum 100: \n")
dim = int(dim)    

render_yn = input("Do you want to render every episode during data generation? (y/n) ")
while render_yn not in ['y', 'n']:
    render_yn = input("Please provide the answer writing 'y' or 'n': ")
if render_yn is 'y':
    render = True
elif render_yn is 'n':
    render = False

# Data extraction from chosen algorithm
real_act, real_obs, real_term_states, target_task = get_data_from_sim(env=gym.make(domain), model=model_address, dim=dim, render=render, algo=selected_algo)

# Parameters initialization
"""
tau=0.005
tot_sets = 9
epsilons_set = np.zeros((tot_sets,4))
lambda_hyp = 10
list_states_tlambd =[]
list_real_states_tlambd = [] 
eps = 0.001
epsilon = np.zeros((11, 11), float)
np.fill_diagonal(epsilon, eps)


mean_s = np.array([10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],dtype=float)
s_tPl =  np.array([11,11,11,11,11,11,11,11,11,11,11.0],dtype=float)
"""
# Env on which to run CMA
env = gym.make('CustomHopper-source-v0')

#Inizialization of the parameters of the multidim. normal
psi_mean_vector = [10,7,5,3,1,1,1,1]
std_dv = 0.5

parameters_name = ["torsomass", "thighmass", "legmass", "footmass"]

index=0
T = len(real_obs)
lambda_hyp = 10
treshold = T-lambda_hyp
tau=0.005
mse=1


# Outer optimization cycle
while index<treshold and mse>tau:    

    # Initialize CMA optimizer and its options
    opts = cma.CMAOptions()
    
    # Set bounds to avoid unphysical phenomena that may block the execution of the environment
    opts.set("bounds", [[0, 0, 0, 0, 0, 0, 0, 0], [10, 10, 10, 10, 10, 10, 10, 10]])
    
    es = cma.CMAEvolutionStrategy(psi_mean_vector, std_dv, opts) 
    
    # Set stopping criteria of CMA optimizer (inner cycle)
    es.opts.set({'tolfun': 1e-6, 'maxiter': 150})
    
    # Inner optimization cycle
    while not es.stop():
        solutions = es.ask()        
        es.tell(solutions, [likelihood_f(x, real_act, real_obs, real_term_states, env) for x in solutions])
        es.logger.add()
        es.disp()
    
    # Print results
    es.result_pretty()
    psi_mean_vector = es.result[0]
    x = PrettyTable()
    x.field_names = ["Parameter name", "Mean calculated", "Target value", "Std calculated"]
    for i in range(len(target_task)):
        x.add_row([parameters_name[i], psi_mean_vector[i], target_task[i], psi_mean_vector[i+len(target_task)]])
    x.float_format = '.6'
    print(x)

    # Stopping criteria for outer cycle
    mse=np.linalg.norm(target_task - psi_mean_vector[:4])
    print('Squared distance between best means and target task:', mse)
    index=index+1

cma.plot()         




        
        
        
        
#For each parameter csi_k we set the simulator to the original state s_t



#Execution of the actions between t and t+lambda-1



#Computation of s_t+lambda in order to compute the gaussian distribution (use the classical estimators for mean and sigma). Remeber epsilon for sigma. 



#epsilon minimiza MSE between the mean and the s_t+lambda of the true D.



#computation of the log likelihood in order to maximize the likelihood. Use CMA-ES. Parameters are in interval [0,4]



#epsilon: smallest value that give the minimization of MSE. Fix a threshold tau for MSE.



