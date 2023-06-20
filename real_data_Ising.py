## NSDUH 2014 data

# %% import data from tsv file
import pandas as pd
import numpy as np
import numpy.random as npr
import pickle
import os
from depend_funcs import *

## set random seed
np.random.seed(1234)

## read data from the cleaned csv file
df = pd.read_csv('NESARC_cleaned_unknownNA_no_com_mis.csv', sep=',', index_col=0)

## transform data frame to numpy array
data_y_masked = df.to_numpy(dtype=float)
J = 15
print(data_y_masked.shape)

## complete case data
non_mis_indices = np.isnan(data_y_masked).sum(1) == 0
data_y_complete = data_y_masked[non_mis_indices, :]


# %% run estimation program

# Set hyper-parameters
TAU_1_S_SET = 1
MCMC_LEN_SET = 10000
K0_SET = 10

# Set repetitions number from environment variable
# reps = int(os.getenv('SLURM_ARRAY_TASK_ID'))
reps = 1
np.random.seed(reps)

# Generate initial S0 matrix
S0 = npr.uniform(-0.1, 0.1, (J, J))
S0 = np.triu(S0) + np.triu(S0, 1).T

# Run mis_ising function on complete data
alpha_res_complete_analysis = mis_ising(data_y_complete.copy(), S0.copy(), TAU_1_S_SET, k0=K0_SET, mcmc_len=MCMC_LEN_SET)
print("complete_analysis finished.")

# Run mis_ising function on proposed data
alpha_res_proposed = mis_ising(data_y_masked.copy(), S0.copy(), TAU_1_S_SET, k0=K0_SET, mcmc_len=MCMC_LEN_SET)
print("proposed finished.")

# Save results to a pickle file
with open(f'real_data_res/res_unknownNA_no_com_mis_{reps}.pkl', 'wb') as f:
    pickle.dump([alpha_res_proposed, alpha_res_complete_analysis], f)

