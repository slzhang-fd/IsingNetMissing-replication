import numpy as np
import numpy.random as npr
from polyagamma import random_polyagamma
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def my_norm_pdf(x, loc=0.0, scale=1.0):
    """Numerically stable normal pdf function."""
    return np.exp(-((x - loc) ** 2) / (2 * scale**2)) / np.sqrt(2 * np.pi) / scale


## data generation function from the true model
def generate_y_theta(A, S, N, mcmc_len=1000, silent=True):
    ## Because of the non-standard form of prior theta in the model,
    ## we generate data y from a mcmc process that iterative
    ## between sampling y and theta given each other
    ## use Theta_0 from standard normal
    J, K = A.shape
    theta_all = npr.normal(0, 1, size=(N, K))
    y_prob = sigmoid(theta_all @ A.T)
    y_all = (npr.random((N, J)) < y_prob).astype(float)
    theta0_chains = []
    for i in range(mcmc_len):
        if i % 100 == 0 and not silent:
            print("\rIteration: {}".format(i))
        ## sample y_j given y_{-j} and theta
        for j in range(J):
            y_all_j = y_all.copy()
            y_all_j[:, j] = 0.5
            c_j = y_all_j @ S[:, j]
            y_prob_j = sigmoid(c_j + theta_all @ A[j, :])
            y_all[:, j] = (npr.random(N) < y_prob_j).astype(float)

        ## sample theta given y_all
        theta_all = y_all @ A + npr.normal(0, 1, size=(N, K))
        ## record theta chain
        theta0_chains.append(theta_all[0, :])

    return y_all, theta_all, theta0_chains


## The created duplication matrix will convert vech(S) to vec(S)
## vech(S)
## Examples:
## (duplication_matrix_matlab(J).todense() @ range(1,int(J*(J+1)/2+1))).reshape(J,J)
## (duplication_matrix_matlab(J).todense() @ S[np.triu_indices(J)]).reshape(J,J)
def duplication_matrix(n):
    m = n * (n + 1) // 2
    nsq = n**2
    r = 1
    a = 1
    v = np.zeros(nsq, dtype=int)

    for i in range(1, n + 1):
        b = i
        for j in range(i - 1):
            v[r - 1] = b
            b = b + n - j - 1
            r = r + 1

        for j in range(n - i + 1):
            v[r - 1] = a + j
            r = r + 1

        a = a + n - i + 1

    D2 = csr_matrix((np.ones(nsq), (np.arange(nsq), v - 1)), shape=(nsq, m))

    return D2  # .todense()


def randomly_mask(Y, p):
    mask = np.random.rand(*Y.shape) < p
    Y_masked = np.where(mask, np.nan, Y)
    return Y_masked


def MAR_mask(Y, p, q):
    # mask = []
    Y_masked = Y.copy().astype(float)
    for j in range(5):
        mask = np.random.rand(Y.shape[0]) < (p[j] * Y[:, -1] + q[j] * (1 - Y[:, -1]))
        Y_masked[mask, j] = np.nan

    return Y_masked


def screening_mask(Y):
    # mask = []
    Y_masked = Y.copy().astype(float)
    # set columns 2 to the end as nan if the first two columns are all 0
    for j in range(2, Y.shape[1]):
        Y_masked[:, j] = np.where((Y[:, 0] == 0) & (Y[:, 1] == 0), np.nan, Y[:, j])

    return Y_masked


## Ising model with missing data
## The goal is to estimate S and edge non-zero indicator probabilities
def mis_ising(data_y, S, tau_1_sq, k0, mcmc_len=1000, silent=True):
    ## set default values
    N, J = data_y.shape
    # tau_1_S = 1
    # tau_0_sq = tau_1_sq / (C_const * N * np.log(N))
    # tau_0 = np.sqrt(tau_0_sq)
    # tau_1 = np.sqrt(tau_1_sq)
    tau_2_sq = 100
    ## set initial values for latent variables
    ## impute random values in the missing entries of data_y
    assert data_y.dtype == float, "Array is not of type float"
    missing_indices = np.isnan(data_y)
    data_y[missing_indices] = np.random.randint(2, size=missing_indices.sum())
    # delta_S = S > 0  # .astype(int)

    ## generate matrix M
    T = duplication_matrix(J)  # J^2 x J(J+1)/2
    M = np.zeros((N * J, int(J * (J + 1) / 2)))  # N*J x J(J+1)/2

    alpha_chains = []
    # S_chains = []
    S_auxillary = S.copy()
    D_s = np.ones((J, J)) / tau_1_sq
    np.fill_diagonal(D_s, 1.0 / tau_2_sq)
    D_s_D = np.diag(D_s.T.reshape(-1))
    # Create a progress bar with tqdm
    with tqdm(total=mcmc_len, desc="MCMC Progress", ncols=80, disable=silent) as pbar:
        for iter in range(mcmc_len):
            ## 0. sample y and replace the missing entries
            for j in range(J):
                Y_j_star = data_y.copy()
                Y_j_star[:, j] = 0.5
                Phi_j = Y_j_star @ S_auxillary[:, j]
                y_prob_j = sigmoid(Phi_j)
                tmp_y_j = npr.random(N) < y_prob_j  ## .astype(int)
                ## replace in-place the missing entries of j-th column
                data_y[:, j][missing_indices[:, j]] = tmp_y_j[missing_indices[:, j]]

                ## update delta_sj
                # tmp1_sj = my_norm_pdf(S_auxillary[:,j], loc=0, scale=tau_1)
                # tmp0_sj = my_norm_pdf(S_auxillary[:,j], loc=0, scale=tau_0)
                # delta_sj_prob = sparse_s * tmp1_sj / (sparse_s * tmp1_sj + (1-sparse_s) * tmp0_sj)
                # delta_sj = (npr.random(J) < delta_sj_prob)

                ## update omega_sj
                # D_sj = (1-delta_sj) / tau_0_sq + delta_sj / tau_1_sq
                D_sj = np.ones(J) / tau_1_sq
                D_sj[j] = 1.0 / tau_2_sq
                D_sj_D = np.diag(D_sj)
                Omega_j = random_polyagamma(np.ones(N), Phi_j)
                Omega_j_D = diags(Omega_j)

                ## update S_auxillary_j
                cho_fsj = cho_factor(Y_j_star.T @ Omega_j_D @ Y_j_star + D_sj_D)
                Sigma_sj = cho_solve(cho_fsj, np.eye(int(J)))
                Kappa_j = data_y[:, j] - 0.5
                mu_sj = Sigma_sj @ Y_j_star.T @ Kappa_j
                S_auxillary[:, j] = npr.multivariate_normal(mu_sj, Sigma_sj)

            if iter % int(k0) == 0:
                ## 1. sample delta_S, the sparsity prior
                # tmp1_S = my_norm_pdf(S, loc=0, scale=tau_1)
                # tmp0_S = my_norm_pdf(S, loc=0, scale=tau_0)
                # delta_S_prob = sparse_s * tmp1_S / (sparse_s * tmp1_S + (1-sparse_s) * tmp0_S)
                # delta_S = (npr.random(S.shape) < delta_S_prob)#.astype(int)
                # ## make delta_S symmetric
                # delta_S = np.triu(delta_S) + np.triu(delta_S, 1).T
                # np.fill_diagonal(delta_S, 0)

                ## 2. sample S
                ## We incorperate the symmetric structure of S
                for tt in range(5):
                    Phi = data_y @ S + (-data_y + 0.5) * np.outer(
                        np.ones(N), np.diag(S)
                    )
                    Omega = random_polyagamma(np.ones((N, J)), Phi)
                    # Omega_D = np.diag(Omega.T.reshape(-1))
                    Omega_D = diags(Omega.T.reshape(-1))

                    # print(M.shape, Omega_D.shape, T.shape)
                    # Sigma_alpha = inv(M.T @ Omega_D @ M + T.T @ D_s_D @ T)
                    for j in range(J):
                        Y_j_star = data_y.copy()
                        Y_j_star[:, j] = 0.5
                        M[j * N : (j + 1) * N, :] = Y_j_star @ T[j * J : (j + 1) * J, :]

                    Kappa = data_y - 0.5
                    cho_fs = cho_factor(M.T @ Omega_D @ M + T.T @ D_s_D @ T)
                    Sigma_alpha = cho_solve(cho_fs, np.eye(int(J * (J + 1) / 2)))
                    mu_alpha = Sigma_alpha @ M.T @ Kappa.T.reshape(-1)
                    alpha = npr.multivariate_normal(mu_alpha, Sigma_alpha)
                    S[np.triu_indices(J)] = alpha
                    S = np.triu(S) + np.triu(S, 1).T

                ## record MCMC chains of S
                alpha_chains.append(alpha.copy())
                # S_chains.append(S.copy())

            ## Update the progress bar
            pbar.update(1)

        return alpha_chains
