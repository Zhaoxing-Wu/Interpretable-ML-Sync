import networkx as nx
import numpy as np
from tqdm import trange


def generate_nxg(X, verbose=True):
    """Generate a list of networkx graphs from the provided list of vectorized adjacency matrices
    
    Args:
        X (array): List of vectorized adjacency matrices in the form of np.ndarray
        verbose (bool, optional): Whether to print progress. Defaults to True.

    Returns:
        graph_list (array): List of networkx.Graph generated from the adjacency matrices
    """
    graph_list = []
    k = int(np.sqrt(X.shape[0]))
    for i in range(X.shape[1]):
        adj_mat = X.T[i].reshape(k,k)
        G = nx.from_numpy_matrix(adj_mat)
        graph_list.append(G)
    
    if verbose:
        print('Generated {} graphs'.format(len(graph_list)))
    return graph_list

def coding(X, W, H0, 
          r=None, 
          a1=0, #L1 regularizer
          a2=0, #L2 regularizer
          sub_iter=[5], 
          stopping_grad_ratio=0.0001, 
          nonnegativity=True,
          subsample_ratio=1):
    """ Find hat{H} = argmin_H ( || X - WH||_{F}^2 + a1*|H| + a2*|H|_{F}^{2} ) 
    within radius r from H0. Use row-wise projected gradient descent

    Args:
        X (array): Data matrix
        W (array): Dictionary matrix
        H0 (array): Initial guess of H
        r (float, optional): Radius of the ball. Defaults to None.
        a1 (int, optional): L1 regularizer. Defaults to 0.
        a2 (int, optional): L2 regularizer. Defaults to 0.
        sub_iter (list, optional): Number of iterations for each subproblem. Defaults to [5].
        stopping_grad_ratio (float, optional): Stopping criteria. Defaults to 0.0001.
        nonnegativity (bool, optional): Whether to enforce nonnegativity. Defaults to True.
        subsample_ratio (float, optional): Ratio of the data to be used. Defaults to 1.

    Returns:
        H (array): Optimal H
    """
    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[1])
    if subsample_ratio>1:  # subsample columns of X and solve reduced problem (like in SGD)
        idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
    A = W.T @ W ## Needed for gradient computation
    grad = W.T @ (W @ H0 - X)
    while (i < np.random.choice(sub_iter)):
        step_size = (1 / (((i + 1) ** (1)) * (np.trace(A) + 1)))
        H1 -= step_size * grad 
        if nonnegativity:
            H1 = np.maximum(H1, 0)  # nonnegativity constraint
        i = i + 1
    return H1

def ALS(X,
        n_components = 10, # number of columns in the dictionary matrix W
        n_iter=100,
        a0 = 0, # L1 regularizer for H
        a1 = 0, # L1 regularizer for W
        a12 = 0, # L2 regularizer for W
        H_nonnegativity=True,
        W_nonnegativity=True,
        compute_recons_error=False,
        subsample_ratio = 1):
        """Alternating Least Squares for NMF.
        Given data matrix X, use alternating least squares to find factors W,H so that 
                                || X - WH ||_{F}^2 + a0*|H|_{1} + a1*|W|_{1} + a12 * |W|_{F}^{2}
        is minimized (at least locally).

        Args:
            X (array): Data matrix
            n_components (int, optional): Number of columns in the dictionary matrix W. Defaults to 10.
            n_iter (int, optional): Number of iterations. Defaults to 100.
            a0 (int, optional): L1 regularizer for H. Defaults to 0.
            a1 (int, optional): L1 regularizer for W. Defaults to 0.
            a12 (int, optional): L2 regularizer for W. Defaults to 0.
            H_nonnegativity (bool, optional): Whether to enforce nonnegativity on H. Defaults to True.
            W_nonnegativity (bool, optional): Whether to enforce nonnegativity on W. Defaults to True.
            compute_recons_error (bool, optional): Whether to compute reconstruction error. Defaults to False.
            subsample_ratio (float, optional): Ratio of the data to be used. Defaults to 1.

        Returns:
            W (array): Dictionary matrix
            H (array): Coefficient matrix
            recons_error (float): Reconstruction error
        """
        
        d, n = X.shape
        r = n_components
        W = np.random.rand(d,r)
        H = np.random.rand(r,n) 
        
        for i in trange(n_iter):
            H = coding(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            W = coding(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity,
                       subsample_ratio=subsample_ratio).T
            W /= np.linalg.norm(W)
            if compute_recons_error and (i % 10 == 0) :
                print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H)**2))
        return W, H