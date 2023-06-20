# %% 
#####################################################
## Fifteen-node simulation in the paper:
## A Note on Ising Network Analysis with Missing Data
#####################################################

# Import necessary libraries
import os
import pickle

import numpy as np
import numpy.random as npr
import pandas as pd

## Import functions from depend_funcs.py
from depend_funcs import *

## Set the seed for reproducibility
np.random.seed(1)

## Define sample size, number of variables
J = 15

## Set a sparse network structure of S
S = np.zeros((J, J))
S_sparsity = 0.3

## randomly select some elements in the lower triangular matrix to be non-zero
S[np.tril_indices(J,-1)] = (npr.uniform(0, 1, size=int(J*(J-1)/2)) < S_sparsity) * npr.uniform(0.4, 1, size=int(J*(J-1)/2)) * npr.choice([-1, 1], size=int(J*(J-1)/2))

## make S symmetric
S = S + S.T

## The diagonal of S is s_{jj} + 2*b_j, we set to be 0 here
np.fill_diagonal(S, 0)

#%%

## hyper-parameters setting
TAU_1_S_SET = 1
MCMC_LEN_SET = 5000
BURN_IN_SET = 1000
K0_SET = 10

# Define the sample sizes to be used in the simulation
N = [1000, 2000, 4000, 8000]

# Create an empty list to store the results of the simulation
res_all_plain = []

# Set repetitions number from environment variable
# reps = int(os.getenv('SLURM_ARRAY_TASK_ID'))
reps = 1
np.random.seed(reps)

# Loop through each sample size
for i, n in enumerate(N):
    # Print the current sample size and repetition number
    print("N: ", n, "reps: ", reps)
    
    ## generate data from the true model
    # Generate data from the true model with the given parameters
    data_y, data_theta, theta0_chain = generate_y_theta(np.zeros((J, 1)), S, n, mcmc_len=1000, silent=True)
    
    # Randomly mask half of the data
    data_y_masked = randomly_mask(data_y, 0.5)

    # Initialize the starting value for the S matrix
    S0 = npr.uniform(-0.1,0.1,(J,J))
    S0 = np.triu(S0) + np.triu(S0, 1).T
    
    # Fit the MIS-Ising model to the masked data
    alpha_res = mis_ising(data_y_masked.copy(), S0.copy(), tau_1_sq=TAU_1_S_SET,
                                  k0=K0_SET, mcmc_len=MCMC_LEN_SET, silent=True)
    
    # Store the results in a dataframe and append it to the list of results
    res_all_plain.append(pd.DataFrame({'alpha_res': np.mean(alpha_res[int(BURN_IN_SET/K0_SET):], axis=0), 
                                    'S': S[np.triu_indices(J)]}))
    print("spec plain finished.")


#%%
## save results
with open('simu2_res/simu2_res_rep'+str(reps)+'.pkl', 'wb') as f:
    pickle.dump(res_all_plain, f)
