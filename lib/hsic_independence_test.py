import torch
import sigkernel

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

def get_hisc_null_dist_mc_approx(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    rng = torch.Generator()
    rng.manual_seed(1234)
    
    hist = []
    
    for i in range(no_shuffles):
        idx = torch.randperm(Y.shape[0], generator = rng)
        gram_x, gram_y = get_gram_matrices(X, Y[idx, :, :], static_kernel = sigkernel.LinearKernel())
        hsic_score = get_hsic_score(gram_x, gram_y)
        
        hist.append(hsic_score.item())
        
    return torch.tensor(sorted(hist))

def get_test_result(X, Y, dyadic_order = 1, static_kernel = sigkernel.RBFKernel(sigma=0.5), no_shuffles = 50):
    gram_x, gram_y = get_gram_matrices(X, Y, dyadic_order, static_kernel)
    hsic_score = get_hsic_score(gram_x, gram_y)
    null_pdf_empirical = get_hsic_null_dist_mc_approx(X, Y, dyadic_order, static_kernel, no_shuffles)

    p_value = (null_pdf_empirical > hsic_score).sum()/len(null_pdf_empirical)

    return hsic_score.item(), p_value.item()
