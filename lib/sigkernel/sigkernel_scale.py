import numpy as np
import torch
import torch.cuda
from numba import cuda

from cython_backend import sig_kernel_batch_varpar, sig_kernel_Gram_varpar
from cuda_backend import compute_sig_kernel_batch_varpar_from_increments_cuda, compute_sig_kernel_Gram_mat_varpar_from_increments_cuda


# ===========================================================================================================
# Static kernels
# ===========================================================================================================
class StaticKernel():
    def batch_kernel(self, X, Y):
        raise NotImplementedError()

    def Gram_matrix(self, X, Y):
        raise NotImplementedError()

    def post_scaled_batch_kernel(self, X, Y, post_scale_X, post_scale_Y):
        assert X.shape[0] == len(post_scale_X), 'post_scale_X ({}) needs to be equal to batch size ({})'.format(X.shape[0], len(post_scale_X))
        assert Y.shape[0] == len(post_scale_Y), 'post_scale_Y ({}) needs to be equal to batch size ({})'.format(Y.shape[0], len(post_scale_Y))

        expanded_post_scale_X = torch.tile(post_scale_X, (X.shape[1], 1)) # length_X x batch
        expanded_post_scale_Y = torch.tile(post_scale_Y, (Y.shape[1], 1)) # length_Y x batch

        batch_kernel = self.batch_kernel(X, Y)
        scale = torch.einsum('ji,ki->ijk', expanded_post_scale_X, expanded_post_scale_Y)

        return batch_kernel * scale
        

    def post_scaled_gram_matrix(self, X, Y, post_scale_X, post_scale_Y):
        assert X.shape[0] == len(post_scale_X), 'post_scale_X ({}) needs to be equal to batch size ({})'.format(X.shape[0], len(post_scale_X))
        assert Y.shape[0] == len(post_scale_Y), 'post_scale_Y ({}) needs to be equal to batch size ({})'.format(Y.shape[0], len(post_scale_Y))

        expanded_post_scale_X = torch.tile(post_scale_X, (X.shape[1], 1)) # length_X x batch_X
        expanded_post_scale_Y = torch.tile(post_scale_Y, (Y.shape[1], 1)) # length_Y x batch_Y

        gram = self.Gram_matrix(X, Y)
        scale = torch.einsum('ij,kl->jlik', expanded_post_scale_X, expanded_post_scale_Y)

        return gram * scale


class LinearKernel(StaticKernel):
    """Linear kernel k: R^d x R^d -> R"""

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        return torch.bmm(X, Y.permute(0,2,1))

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        return torch.einsum('ipk,jqk->ijpq', X, Y)


class RBFKernel(StaticKernel):
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, sigma):
        self.sigma = sigma

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))
        return torch.exp(-dist/self.sigma)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))
        return torch.exp(-dist/self.sigma)
# ===========================================================================================================



# ===========================================================================================================
# Signature Kernel
# ===========================================================================================================
class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self,static_kernel, dyadic_order, _naive_solver=False):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def compute_kernel(self, X, Y, post_scale_static_x = None, post_scale_static_y = None):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector k(X^i_T,Y^i_T) of shape (batch,)
        """
        return _SigKernel.apply(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver, post_scale_static_x, post_scale_static_y)

    def compute_Gram(self, X, Y, post_scale_static_x = None, post_scale_static_y = None, sym = False):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """
        return _SigKernelGram.apply(X, Y, self.static_kernel, self.dyadic_order, sym, self._naive_solver, post_scale_static_x, post_scale_static_y)

    def compute_distance(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """
        
        assert not Y.requires_grad, "the second input should not require grad"

        k_XX = self.compute_kernel(X, X)
        k_YY = self.compute_kernel(Y, Y)
        k_XY = self.compute_kernel(X, Y)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2.*torch.mean(k_XY) 

    def compute_mmd(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True)
        K_YY = self.compute_Gram(Y, Y, sym=True)
        K_XY = self.compute_Gram(X, Y, sym=False)

        return torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)


class _SigKernel(torch.autograd.Function):
    """Signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""
 
    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, _naive_solver=False, post_scale_static_x = None, post_scale_static_y = None):

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^i_t)
        if post_scale_static_x is None:
            G_static = static_kernel.batch_kernel(X,Y)
        else:
            G_static = static_kernel.post_scaled_batch_kernel(X,Y, post_scale_static_x, post_scale_static_y)

        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM+1,NN+1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'
            
            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            K[:,0,:] = 1.
            K[:,:,0] = 1. 

            # Compute the forward signature kernel
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K), _naive_solver)
            K = K[:,:-1,:-1]

        # if on CPU
        else:
            K = torch.tensor(sig_kernel_batch_varpar(G_static_.detach().numpy(), _naive_solver), dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X,Y,G_static,K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return K[:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):
    
        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=1),dim=2)

        # if on GPU
        if X.device.type=='cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM+2, NN+2), device=G_static_rev.device, dtype=G_static_rev.dtype) 
            K_rev[:,0,:] = 1.
            K_rev[:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM,NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K_rev), _naive_solver)

            K_rev = K_rev[:,:-1,:-1]      

        # if on CPU
        else:
            K_rev = torch.tensor(sig_kernel_batch_varpar(G_static_rev.detach().numpy(), _naive_solver), dtype=G_static.dtype, device=G_static.device)

        K_rev = flip(flip(K_rev,dim=1),dim=2)
        KK = K[:,:-1,:-1] * K_rev[:,1:,1:]   
        
        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D, dtype=X.dtype, device=X.device)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.batch_kernel(Xh,Y) 
        G_h = G_h.reshape(A,M,D,N)
        G_h = G_h.permute(0,1,3,2) 

        Diff_1 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_1 =  tile( tile(Diff_1,1,2**dyadic_order)/float(2**dyadic_order),2, 2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_2 += - G_h[:,:-1,1:,:] + G_h[:,:-1,:-1,:] + (G_static[:,:-1,1:])[:,:,:,None] - (G_static[:,:-1,:-1])[:,:,:,None]
        Diff_2 = tile(tile(Diff_2,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (KK[:,:,:,None] * Diff_1)/h
        grad_2 = (KK[:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=2)
        grad_1 = torch.sum(grad_1.reshape(A,M-1,2**dyadic_order,D),axis=2)
        grad_2 = torch.sum(grad_2,axis=2)
        grad_2 = torch.sum(grad_2.reshape(A,M-1,2**dyadic_order,D),axis=2)

        grad_prev = grad_1[:,:-1,:] + grad_2[:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:,1:,:]],dim=1)   # /
        grad_incr = grad_prev - grad_1[:,1:,:]
        grad_points = torch.cat([(grad_2[:,0,:]-grad_1[:,0,:])[:,None,:],grad_incr,grad_1[:,-1,:][:,None,:]],dim=1)

        if Y.requires_grad:
            grad_points*=2

        return grad_output[:,None,None]*grad_points, None, None, None, None


class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, sym=False, _naive_solver=False, post_scale_static_x = None, post_scale_static_y = None):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^j_t)
        if post_scale_static_x is None:
            G_static = static_kernel.Gram_matrix(X,Y)
        else:
            G_static = static_kernel.post_scaled_gram_matrix(X,Y, post_scale_static_x, post_scale_static_y)

        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM,NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G[:,:,0,:] = 1.
            G[:,:,:,0] = 1. 

            # Run the CUDA kernel.
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G), _naive_solver)

            G = G[:,:,:-1,:-1]

        else:
            G = torch.tensor(sig_kernel_Gram_varpar(G_static_.detach().numpy(), sym, _naive_solver), dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X,Y,G,G_static)      
        ctx.sym = sym
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return G[:,:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G, G_static = ctx.saved_tensors
        sym = ctx.sym
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=2),dim=3)

        # if on GPU
        if X.device.type=='cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            G_rev = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G_rev[:,:,0,:] = 1.
            G_rev[:,:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G_rev), _naive_solver)

            G_rev = G_rev[:,:,:-1,:-1]

        # if on CPU
        else:
            G_rev = torch.tensor(sig_kernel_Gram_varpar(G_static_rev.detach().numpy(), sym, _naive_solver), dtype=G_static.dtype, device=G_static.device)

        G_rev = flip(flip(G_rev,dim=2),dim=3)
        GG = G[:,:,:-1,:-1] * G_rev[:,:,1:,1:]     

        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D, dtype=X.dtype, device=X.device)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.Gram_matrix(Xh,Y) 
        G_h = G_h.reshape(A,B,M,D,N)
        G_h = G_h.permute(0,1,2,4,3) 

        Diff_1 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_1 =  tile(tile(Diff_1,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_2 += - G_h[:,:,:-1,1:,:] + G_h[:,:,:-1,:-1,:] + (G_static[:,:,:-1,1:])[:,:,:,:,None] - (G_static[:,:,:-1,:-1])[:,:,:,:,None]
        Diff_2 = tile(tile(Diff_2,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (GG[:,:,:,:,None] * Diff_1)/h
        grad_2 = (GG[:,:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=3)
        grad_1 = torch.sum(grad_1.reshape(A,B,M-1,2**dyadic_order,D),axis=3)
        grad_2 = torch.sum(grad_2,axis=3)
        grad_2 = torch.sum(grad_2.reshape(A,B,M-1,2**dyadic_order,D),axis=3)

        grad_prev = grad_1[:,:,:-1,:] + grad_2[:,:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_1[:,:,1:,:]], dim=2)   # /
        grad_incr = grad_prev - grad_1[:,:,1:,:]
        grad_points = torch.cat([(grad_2[:,:,0,:]-grad_1[:,:,0,:])[:,:,None,:],grad_incr,grad_1[:,:,-1,:][:,:,None,:]],dim=2)

        if sym:
            grad = (grad_output[:,:,None,None]*grad_points + grad_output.t()[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None
        else:
            grad = (grad_output[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None
# ===========================================================================================================



# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================



# ===========================================================================================================
# Hypothesis test functionality
# ===========================================================================================================
def c_alpha(m, alpha):
    return 4. * np.sqrt(-np.log(alpha) / m)

def hypothesis_test(y_pred, y_test, static_kernel, confidence_level=0.99, dyadic_order=0):
    """Statistical test based on MMD distance to determine if 
       two sets of paths come from the same distribution.
    """

    k_sig = SigKernel(static_kernel, dyadic_order)

    m = max(y_pred.shape[0], y_test.shape[0])
    
    TU = k_sig.compute_mmd(y_pred,y_test)  
  
    c = torch.tensor(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(f'Hypothesis rejected: distribution are not equal with {confidence_level*100}% confidence')
    else:
        print(f'Hypothesis accepted: distribution are equal with {confidence_level*100}% confidence')
# ===========================================================================================================








# ===========================================================================================================
# Deprecated implementation (just for testing)
# ===========================================================================================================
def SigKernel_naive(X, Y, static_kernel, dyadic_order=0, _naive_solver=False):

    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2**dyadic_order)*(M-1)
    NN = (2**dyadic_order)*(N-1)

    K_XY = torch.zeros((A, MM+1, NN+1), dtype=X.dtype, device=X.device)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^i_t)
    G_static = static_kernel.batch_kernel(X,Y)
    G_static = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
    G_static = tile(tile(G_static,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:,i,j].clone()

            k_10 = K_XY[:, i + 1, j].clone()
            k_01 = K_XY[:, i, j + 1].clone()
            k_00 = K_XY[:, i, j].clone()

            if _naive_solver:
                K_XY[:, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            else:
                K_XY[:, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
                #K_XY[:, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)
            
    return K_XY[:, -1, -1]


class SigLoss_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigLoss_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self,X,Y):

        k_XX = SigKernel_naive(X,X,self.static_kernel,self.dyadic_order,self._naive_solver)
        k_YY = SigKernel_naive(Y,Y,self.static_kernel,self.dyadic_order,self._naive_solver)
        k_XY = SigKernel_naive(X,Y,self.static_kernel,self.dyadic_order,self._naive_solver)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2.*torch.mean(k_XY)


def SigKernelGramMat_naive(X,Y,static_kernel,dyadic_order=0,_naive_solver=False):

    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2**dyadic_order)*(M-1)
    NN = (2**dyadic_order)*(N-1)

    K_XY = torch.zeros((A,B, MM+1, NN+1), dtype=X.dtype, device=X.device)
    K_XY[:,:, 0, :] = 1.
    K_XY[:,:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^j_t)
    G_static = static_kernel.Gram_matrix(X,Y)
    G_static = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
    G_static = tile(tile(G_static,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:,:,i,j].clone()

            k_10 = K_XY[:, :, i + 1, j].clone()
            k_01 = K_XY[:, :, i, j + 1].clone()
            k_00 = K_XY[:, :, i, j].clone()

            if _naive_solver:
                K_XY[:, :, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            else:
                K_XY[:, :, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
                #K_XY[:, :, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)

    return K_XY[:,:, -1, -1]


class SigMMD_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigMMD_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self, X, Y):

        K_XX = SigKernelGramMat_naive(X,X,self.static_kernel,self.dyadic_order,self._naive_solver)
        K_YY = SigKernelGramMat_naive(Y,Y,self.static_kernel,self.dyadic_order,self._naive_solver)  
        K_XY = SigKernelGramMat_naive(X,Y,self.static_kernel,self.dyadic_order,self._naive_solver)
        
        return torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY) 
