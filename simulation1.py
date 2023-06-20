# %% 
#####################################################
## Six-node simulation in the paper:
## A Note on Ising Network Analysis with Missing Data
#####################################################

# Import necessary libraries
import os
import pickle

import numpy as np
import numpy.random as npr
import pandas as pd

# Import functions from depend_funcs.py
from depend_funcs import *

# Set the seed for reproducibility
np.random.seed(1234)

# Define sample size, number of variables
J = 6

# Set a sparse network structure of S
S = np.zeros((J, J))
S_sparsity = 0.5

# Randomly select some elements in the lower triangular matrix to be non-zero
indices = np.tril_indices(J,-1)
S[indices] = (npr.uniform(0, 1, size=int(J*(J-1)/2)) < S_sparsity) * npr.uniform(0.4, 1, size=int(J*(J-1)/2)) * npr.choice([-1, 1], size=int(J*(J-1)/2))

# Make S symmetric
S = S + S.T

# The diagonal of S is s_{jj} + 2*b_j, we set it to be 0 here
np.fill_diagonal(S, 0)

#%%
# Set hyper-parameters
TAU_1_S_SET = 1
MCMC_LEN_SET = 5000
BURN_IN_SET = 1000
K0_SET = 10
N = [1000, 2000, 4000, 8000]
P_RATE = [0.3, 0.8, 0.3, 0.7, 0.2]
Q_RATE = [0.8, 0.5, 0.7, 0.4, 0.9]

# Initialize results lists
res_all_proposed = []
res_all_impute_complete = []
res_all_complete_analysis = []

# Set repetitions number from environment variable
# reps = int(os.getenv('SLURM_ARRAY_TASK_ID'))
reps = 1
np.random.seed(reps)

# Iterate over each N in list
for i, n in enumerate(N):
    print("N: ", n, "reps: ", reps)
    
    # Generate data from the true model
    data_y, data_theta, theta0_chain = generate_y_theta(np.zeros((J, 1)), S, n, mcmc_len=1000, silent=True)
    
    # Apply MAR to mask some elements in data_y
    data_y_masked = MAR_mask(data_y, P_RATE, Q_RATE)
    data_y_complete = data_y_masked[~np.isnan(data_y_masked[:,0]),:]

    # Initialize S0
    S0 = npr.uniform(-0.1,0.1,(J,J))
    S0 = np.triu(S0) + np.triu(S0, 1).T
    
    # Execute and store the results of the proposed method
    alpha_res = mis_ising(data_y_masked, S0.copy(), TAU_1_S_SET, k0=K0_SET, mcmc_len=MCMC_LEN_SET)
    res_all_proposed.append(pd.DataFrame({'alpha_res': np.mean(alpha_res[int(BURN_IN_SET/K0_SET):], axis=0), 'S': S[np.triu_indices(J)]}))
    print("proposed finished,")

    # Execute and store the results of the impute complete method
    alpha_res = mis_ising(data_y_masked, S0.copy(), TAU_1_S_SET, k0=K0_SET, mcmc_len=MCMC_LEN_SET)
    res_all_impute_complete.append(pd.DataFrame({'alpha_res': np.mean(alpha_res[int(BURN_IN_SET/K0_SET):], axis=0), 'S': S[np.triu_indices(J)]}))
    print("impute complete finished,")

    # Execute and store the results of the complete analysis method
    alpha_res = mis_ising(data_y_complete, S0.copy(), TAU_1_S_SET, k0=K0_SET, mcmc_len=MCMC_LEN_SET)
    res_all_complete_analysis.append(pd.DataFrame({'alpha_res': np.mean(alpha_res[int(BURN_IN_SET/K0_SET):], axis=0), 'S': S[np.triu_indices(J)]}))
    print("complete analysis finished.")

#%%
# Save results using pickle
with open('simu1_res/simu1_res_newMAR_rep'+str(reps)+'.pkl', 'wb') as f:
    pickle.dump([res_all_proposed, res_all_impute_complete, res_all_complete_analysis], f)


