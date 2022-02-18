import torch
import scipy.stats as stats

def get_autocov(X, max_lag = 100):
    autocov_est = torch.zeros(max_lag)

    for lag in range(max_lag):
        autocov_est[lag] = (X[lag:] * X[:len(X)-lag]).mean()

    return autocov_est

def get_prod_autocov_under_null(X, Y, max_lag = 100):
    autocov_x = get_autocov(X, max_lag)
    autocov_y = get_autocov(Y, max_lag)

    return autocov_x*autocov_y

def get_asymptotic_variance(X, Y, max_lag = 100):
    prod_autocov_under_null = get_prod_autocov_under_null(X, Y, max_lag)

    return (prod_autocov_under_null.sum() + prod_autocov_under_null[1:].sum())/len(X)

def get_test_stat_dist_under_null(X, Y, max_lag = 100):
    variance = get_asymptotic_variance(X, Y, max_lag)

    return stats.norm(0, torch.sqrt(variance).item())

def get_cov(X, Y, max_lag = 100):
    cov = torch.zeros(max_lag)

    for lag in range(max_lag):
        cov[lag] = (X[lag:] * Y[:len(Y)-lag]).mean()

    return cov

def get_test_result(X, Y, max_lag = 100):
    null_dist = get_test_stat_dist_under_null(X, Y, max_lag)
    cov = get_cov(X, Y, max_lag)

    return cov[0], 1-null_dist.cdf(cov[0])
