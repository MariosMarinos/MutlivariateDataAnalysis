import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
np.set_printoptions(suppress=True)

def Squared_Exponential(X1, X2, multiple_stability, l=0.8, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        multiple_stability : factor to multiple with the identity matrix to have a numerical stability.
    Returns:
        (m x n) matrix.
        ||x - x'||^2 = x^2 + x'^2 - 2 X @ X.T
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
    K = sigma_f * np.exp(-sqdist / (2* l**2))
    # add a bit of noise for numerical stability 
    Noise_K = K + multiple_stability*np.eye(X1.shape[0])
    return Noise_K

def Polynomial_kernel(X1, X2, multiple_stability, degree, alpha = 1):
    """        
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        multiple_stability : factor to multiple with the identity matrix to have a numerical stability.

        Returns the covariance matrix using the polynomial kernel.
    """
    K = alpha * (1 + X1 @ X2.T) ** degree
    Noise_K = K + multiple_stability*np.eye(X1.shape[0])
    return Noise_K

def a(X1, X2, sigma):
    return 2 * X2.T * sigma @ X1

def Neuronal(X1,X2, sigma):
    numerator =a(X1, X2, sigma)
    denominator = np.sqrt((1+a(X1, X1, sigma))*(1+a(X2, X2, sigma)))
    return (2/np.pi) * np.arcsin(numerator / denominator)

def draw_Samples(samples, mu, cov):
    samples = 5
    realizations = np.random.multivariate_normal(mu.ravel(), cov, samples)
    # Plot GP mean, confidence interval and samples 
    for i in range(samples):
        plt.plot(X, realizations[i])
    plt.xlabel('$x$')
    plt.ylabel('$y$')

if __name__ == "__main__":

    # Finite number of points, n x 1 atrix
    # Define constants such as mu matrix, X data and stability term.
    X = np.linspace(-5, 5, 1001).reshape(-1, 1)
    # Mean and covariance of the prior
    mu = np.zeros(X.shape)

    stability_term = 0.0015

    # using squared exponential kernel.

    cov = Squared_Exponential(X, X, stability_term)
    # Draw five samples from the prior.
    plt.style.use('seaborn-bright')
    draw_Samples(5, mu, cov)
    plt.title('Realisations drawn using Squared Exponential kernel.')
    # plt.savefig("example.png")
    # Using polynomial kernel.

    cov = Polynomial_kernel(X, X, stability_term, 2)
    print(cov.shape)
    # Draw five samples from the prior.
    plt.style.use('_classic_test_patch')
    draw_Samples(5, mu, cov)
    plt.title('Realisations drawn using Polynomial kernel with degree level 2.')
    # plt.savefig("example.png")
    sigma = 1.5

    cov = [Neuronal(i, j, sigma) for (i, j) in itertools.product(X, X)]

    cov = np.array(cov).reshape(X.shape[0], X.shape[0])
    # Draw five samples from the prior.
    plt.style.use('_classic_test_patch')
    
    # plt.savefig("example.png")
    draw_Samples(5, mu, cov)
    plt.title('Realisations  drawn using Neuronal kernel with $\Sigma$ = 1.5')
    plt.savefig('Neuronal_kernel.png')
