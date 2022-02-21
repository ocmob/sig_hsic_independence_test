import torch
import sigkernel
from tqdm import tqdm
from itertools import product

def get_gram_matrices(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5)):
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    n = X.shape[0]

    gram_x = torch.zeros(n,n)
    gram_y = torch.zeros(n,n)

    for i, j in product(range(n), range(n)):
        gram_x[i, j] = signature_kernel.compute_kernel(X[i:i+1, :, :], X[j:j+1, :, :])
        gram_y[i, j] = signature_kernel.compute_kernel(Y[i:i+1, :, :], Y[j:j+1, :, :])

    return gram_x, gram_y

def get_hsic_score(gram_x, gram_y):
    n = gram_x.shape[0]
    
    term1 = (gram_x*gram_y).sum()/n**2
    term2 = -2/n**3*(torch.einsum('ij,ik->jk', gram_x, gram_y)).sum()
    term3 = (gram_x*gram_y.sum()).sum()/n**4
    
    return term1+term2+term3

def get_hsic_score_u_statistic(gram_x, gram_y):
    n = gram_x.shape[0]

    zero_diagonal = torch.ones_like(gram_x) - torch.diag_embed(torch.ones(n))
    term1 = (gram_x*gram_y*zero_diagonal).sum()/(n-1)/n

    semi_prod = torch.einsum('ij,ik->ijk', gram_x, gram_y)
    zero_where_min_2_indices_are_equal_3d = (torch.ones_like(semi_prod) 
            - torch.diag_embed(torch.ones(n)).reshape(1, n, n)
            - torch.diag_embed(torch.ones(n)).reshape(n, 1, n)
            - torch.diag_embed(torch.ones(n)).reshape(n, n, 1)
        ) > 0
    term2 = -2/zero_where_min_2_indices_are_equal_3d.sum()*(semi_prod*zero_where_min_2_indices_are_equal_3d).sum()

    full_prod = torch.einsum('ij,kl->ijkl', gram_x, gram_y)
    zero_where_min_2_indices_are_equal_4d = (torch.ones_like(full_prod) 
            - torch.diag_embed(torch.ones(n)).reshape(1, 1, n, n)
            - torch.diag_embed(torch.ones(n)).reshape(1, n, 1, n)
            - torch.diag_embed(torch.ones(n)).reshape(n, 1, 1, n)
            - torch.diag_embed(torch.ones(n)).reshape(1, n, n, 1)
            - torch.diag_embed(torch.ones(n)).reshape(n, 1, n, 1)
            - torch.diag_embed(torch.ones(n)).reshape(n, n, 1, 1)
        ) > 0
    term3 = (full_prod*zero_where_min_2_indices_are_equal_4d).sum()/zero_where_min_2_indices_are_equal_4d.sum()
    
    return term1+term2+term3

def get_hsic_null_dist_mc_approx(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    rng = torch.Generator()
    rng.manual_seed(1234)
    
    hist = []
    
    for i in tqdm(range(no_shuffles)):
        idx = torch.randperm(Y.shape[0], generator = rng)
        gram_x, gram_y = get_gram_matrices(X, Y[idx, :, :], dyadic_order = dyadic_order, static_kernel = static_kernel)
        hsic_score = get_hsic_score(gram_x, gram_y)
        
        hist.append(hsic_score.item())
        
    return torch.tensor(sorted(hist))

def shuffle_gram_matrix(gram_matrix, generator = torch.Generator()):
    perm = torch.randperm(gram_matrix.shape[0], generator = generator)
    shuffled = torch.zeros_like(gram_matrix)

    for j, id1 in enumerate(perm):
        shuffled[j, j] = gram_matrix[id1, id1]

        for k, id2 in enumerate(perm[j+1:]):
            shuffled[j, k+j+1] = gram_matrix[id1, id2]
            shuffled[k+j+1, j] = gram_matrix[id1, id2]

    return shuffled

def get_hsic_null_dist_mc_matrix_shuffle_approx(gram_x, gram_y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    rng = torch.Generator()
    rng.manual_seed(1234)

    hist = []
    
    for i in tqdm(range(no_shuffles)):
        shuffled_gram_y = shuffle_gram_matrix(gram_y, rng)
        hsic_score = get_hsic_score(gram_x, shuffled_gram_y)
        hist.append(hsic_score.item())
        
    return torch.tensor(sorted(hist))

def get_hsic_null_dist_mc_matrix_shuffle_approx_u_stat(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    rng = torch.Generator()
    rng.manual_seed(1234)

    gram_x, gram_y = get_gram_matrices(X, Y, dyadic_order = dyadic_order, static_kernel = static_kernel)
    
    hist = []
    
    for i in tqdm(range(no_shuffles)):
        shuffled_gram_y = shuffle_gram_matrix(gram_y, rng)
        hsic_score = get_hsic_score_u_statistic(gram_x, shuffled_gram_y)
        hist.append(hsic_score.item())
        
    return torch.tensor(sorted(hist))

def get_test_result(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    gram_x, gram_y = get_gram_matrices(X, Y, dyadic_order, static_kernel)
    hsic_score = get_hsic_score(gram_x, gram_y)
    null_pdf_empirical = get_hsic_null_dist_mc_approx(X, Y, dyadic_order, static_kernel, no_shuffles)

    p_value = (null_pdf_empirical > hsic_score).sum()/len(null_pdf_empirical)

    return null_pdf_empirical, hsic_score.item(), p_value.item()

def get_test_result_matrix_shuffle(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    gram_x, gram_y = get_gram_matrices(X, Y, dyadic_order, static_kernel)
    hsic_score = get_hsic_score(gram_x, gram_y)
    null_pdf_empirical = get_hsic_null_dist_mc_matrix_shuffle_approx(gram_x, gram_y, dyadic_order, static_kernel, no_shuffles)

    p_value = (null_pdf_empirical > hsic_score).sum()/len(null_pdf_empirical)

    return null_pdf_empirical, hsic_score.item(), p_value.item()

def get_test_result_matrix_shuffle_u_stat(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50, null_reuse = None):
    gram_x, gram_y = get_gram_matrices(X, Y, dyadic_order, static_kernel)
    hsic_score = get_hsic_score_u_statistic(gram_x, gram_y)

    if null_reuse is not None:
        null_pdf_empirical = null_reuse
    else:
        null_pdf_empirical = get_hsic_null_dist_mc_matrix_shuffle_approx_u_stat(X, Y, dyadic_order, static_kernel, no_shuffles)

    p_value = (null_pdf_empirical > hsic_score).sum()/len(null_pdf_empirical)

    return null_pdf_empirical, hsic_score.item(), p_value.item()
