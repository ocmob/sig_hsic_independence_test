import numpy as np
import torch

## -- GBM PROC GENERATION

def generate_correlated_bms_with_timeline(cov, n_points = 100, rng = default_rng(69)):
    assert 0 <= cov <= 1

    cov_matrix = np.array([[1, cov],[cov, 1]])
    samples = rng.normal(size = (n_points, 2))
    correlated_samples = samples @ cov_matrix

    dt = 1/np.sqrt(n_points)

    bm1 = dt * np.insert(correlated_samples[:, 0].cumsum(), 0, 0.)
    bm2 = dt * np.insert(correlated_samples[:, 1].cumsum(), 0, 0.)

    timeline = np.linspace(0, 1, n_points)

    return bm1[:-1], bm2[:-1], timeline

def generate_correlated_gbms_with_timeline(mu, sigma, cov, n_points = 100, rng = default_rng(69)):
    bm1, bm2, timeline = generate_correlated_bms_with_timeline(cov, n_points, rng)

    gbm1 = np.exp((mu - sigma ** 2 / 2.) * timeline + sigma * bm1)
    gbm2 = np.exp((mu - sigma ** 2 / 2.) * timeline + sigma * bm2)

    return gbm1, gbm2, timeline

def generate_correlated_gbms_piecewise_lin(mu, sigma, cov, n_points = 100, rng = default_rng(69)):
    gbm1, gbm2, timeline = generate_correlated_gbms_with_timeline(mu, sigma, cov, n_points, rng)
    return piecewise_linear_embedding_row_vct(gbm1, timeline), piecewise_linear_embedding_row_vct(gbm2, timeline)

def generate_correlated_gbms_lead_lag(mu, sigma, cov, n_points = 100, rng = default_rng(69)):    
    gbm1, gbm2, timeline = generate_correlated_gbms_with_timeline(mu, sigma, cov, n_points + 1, rng)    
    return lead_lag_embedding_row_vct(gbm1), lead_lag_embedding_row_vct(gbm2)

def generate_gbm_iid_samples(cov, mu = 0.02, sigma = 0.3, genf = generate_correlated_gbms_lead_lag, n_paths = 60, m_time_points = 100):
    a1 = []
    a2 = []

    rng = default_rng(69)

    for i in range(n_paths):
        sig1, sig2 = genf(0.02, 0.3, cov, m_time_points, rng)
        a1.append(sig1)
        a2.append(sig2)
    return torch.Tensor(a1), torch.Tensor(a2)


## -- AR PROC GENERATION

def gen_ar_processes_pair(n, corr, a = 0.8, rng = np.random.default_rng(1234)):
    z = torch.tensor(rng.multivariate_normal([0, 0] , [[1, corr], [corr, 1]], n))
    x = torch.zeros(2, n)
    x[:, 0] = z[0, :]
    a = 0.8
    for i, zi in enumerate(z[1:, :]):
        x[:, i+1] = a*x[:, i] + zi 
        
    return x

def gen_ar_iid_samples_burn_in(n_timesteps, m_samples, corr = 0, a = 0.8, rng = np.random.default_rng(1234)):
    burn_in_time = 10000

    x_short_samples = torch.zeros(m_samples, n_timesteps, 2)

    for i in range(m_samples):
        process = gen_ar_processes(burn_in_time+n_timesteps, corr, a, rng)
        x_short_samples[i, :, :] = process[:, burn_in_time:].T

    return x_short_samples


## -- EMBEDDINGS

def lead_lag_embedding_row_vct(path):
    path = path.reshape(-1, 1)

    return np.concatenate((path[1:], path[:-1]), axis = 1)

def piecewise_linear_embedding_row_vct(path, time):
    assert len(path) == len(time), "Path and Time lengths must be equal"

    path = path.reshape(-1, 1)
    time = time.reshape(-1, 1)

    return np.concatenate((path, time), axis = 1)
