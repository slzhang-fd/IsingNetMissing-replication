# %%
#####################################################
## Six-node simulation in the paper:
## A Note on Ising Network Analysis with Missing Data
#####################################################

# Import necessary libraries
import pickle

import numpy as np
import numpy.random as npr
import pandas as pd

# Import functions from depend_funcs.py
from depend_funcs import (  # noqa: F401
    generate_y_theta,
    mis_ising,
    randomly_mask,
    screening_mask,
)

# Set the seed for reproducibility
# np.random.seed(123)

# Define sample size, number of variables
J = 6

# Set a sparse network structure of S
S = np.zeros((J, J))
S_sparsity = 0.5

# Randomly select some elements in the lower triangular matrix to be non-zero
indices = np.tril_indices(J, -1)
S[indices] = (
    (npr.uniform(0, 1, size=int(J * (J - 1) / 2)) < S_sparsity)
    * npr.uniform(0.5, 1.5, size=int(J * (J - 1) / 2))
    * npr.choice([-1, 1], size=int(J * (J - 1) / 2))
)

# Make S symmetric
S = S + S.T

# The diagonal of S is s_{jj} + 2*b_j, we set it to be 0 here
np.fill_diagonal(S, 0)

# %%
# Set hyper-parameters
TAU_1_S_SET = 1
MCMC_LEN_SET = 5000
BURN_IN_SET = 2000
K0_SET = 10
N = [4000]

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
    data_y, data_theta, theta0_chain = generate_y_theta(
        np.zeros((J, 1)), S, n, mcmc_len=1000, silent=True
    )

    # Apply MAR to mask some elements in data_y
    # data_y_masked = screening_mask(data_y)
    # data_y_complete = data_y_masked[~np.isnan(data_y_masked[:, 0]), :]
    data_y_masked = randomly_mask(data_y, 0.1)
    data_y_complete = data_y_masked[~np.isnan(data_y_masked).any(1), :]

    # Initialize S0
    S0 = npr.uniform(-0.1, 0.1, (J, J))
    S0 = np.triu(S0) + np.triu(S0, 1).T

    # Execute and store the results of the proposed method
    alpha_res = mis_ising(
        data_y_masked.copy(),
        S0.copy(),
        TAU_1_S_SET,
        k0=K0_SET,
        mcmc_len=MCMC_LEN_SET,
        silent=False,
    )
    print("proposed finished,")

    ## estimate S using full data
    data_test = data_y.copy().astype(float)
    # data_test[0, 0] = np.nan
    alpha_res_com = mis_ising(
        data_test.copy(),
        S0.copy(),
        TAU_1_S_SET,
        k0=K0_SET,
        mcmc_len=MCMC_LEN_SET,
        silent=False,
    )
    print("complete all finished,")
    ## estimate S using complete-case data
    alpha_res_com1 = mis_ising(
        data_y_complete.copy(),
        S0.copy(),
        TAU_1_S_SET,
        k0=K0_SET,
        mcmc_len=MCMC_LEN_SET,
        silent=False,
    )
    print("complete finished,")


# %% compare the results
pd.DataFrame(
    {
        "estimated_mis": np.mean(alpha_res[int(BURN_IN_SET / K0_SET) :], axis=0).round(
            2
        ),
        "estimated_complete_all": np.mean(
            alpha_res_com[int(BURN_IN_SET / K0_SET) :], axis=0
        ).round(2),
        "estimated_complete": np.mean(
            alpha_res_com1[int(BURN_IN_SET / K0_SET) :], axis=0
        ).round(2),
        "true": S[np.triu_indices(J)].round(2),
    }
)

# %%
# Save results using pickle
with open("simu1_res/simu1_res_newMAR_rep" + str(reps) + ".pkl", "wb") as f:
    pickle.dump(
        [res_all_proposed, res_all_impute_complete, res_all_complete_analysis], f
    )
