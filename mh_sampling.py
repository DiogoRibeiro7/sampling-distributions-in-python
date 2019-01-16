import numpy as np
import matplotlib.pyplot as plt


def numpy_gaussian(mu, sigma, nums):
    dims = mu.shape[0]
    # X = N0 * Sigma^0.5 + Mu
    return np.dot(np.random.randn(nums, dims), np.linalg.cholesky(sigma)) + mu


def single_gaussian_pdf(xs, mu, sigma):

    m = xs.shape[0]        
    pxs = np.zeros(m)      
    n = mu.shape[0]        
    
    frac = ((2 * np.pi) ** (n/2)) * np.sqrt(np.linalg.det(np.mat(sigma)))

    for i in range(m):
        ux = xs[i, :] - mu
        px = np.exp(-0.5 * np.dot(np.dot(ux.reshape(1, n), np.mat(sigma).I.A), ux.reshape(n, 1)))/frac
        pxs[i] = px
    return pxs



def mix_gaussian_pdf(xs, alphas, mus, sigmas):

    K = len(alphas)               
    m = xs.shape[0]               
    
    pxs = np.zeros(m)
    for k in range(K):
        alpha = alphas[k]
        mu = mus[k]
        sigma = sigmas[k]
        pxs += alpha * single_gaussian_pdf(xs, mu, sigma)
    return pxs


def mh_sampling():
    steps1 = 2000        
    steps2 = 8000        
    xn1 = 0.0        
    

    mu = np.array([0.0])
    sigma = np.array([[10.0]])
    

    alphas = [0.5, 0.27, 0.23]        
    mus = [np.array([0.0]), np.array([-3.0]), np.array([3.0])]           
    sigmas = [np.array([[1.0]]), np.array([[2.0]]), np.array([[0.5]])]   
    

    nums = []
    for i in range(steps1 + steps2):
        xn = numpy_gaussian(mu = mu, sigma = sigma, nums = 1)[0]                
        pxn1 = mix_gaussian_pdf(np.array([[xn1]]), alphas, mus, sigmas)[0]      
        pxn = mix_gaussian_pdf(np.array([[xn]]), alphas, mus, sigmas)[0]        
        q_xn1_xn = single_gaussian_pdf(np.array([[xn1]]), mu, sigma)[0]         
        q_xn_xn1 = single_gaussian_pdf(np.array([[xn]]), mu, sigma)[0]          
        
        u = np.random.rand()
        if u <= min(1, (pxn * q_xn_xn1) / (pxn1 * q_xn1_xn)):
            xn1 = xn
        if i > steps1:
            nums.append(xn1)
    

    xs = np.linspace(-5, 5, 3000)
    mix_y = mix_gaussian_pdf(xs.reshape((-1, 1)), alphas = alphas, mus = mus, sigmas = sigmas)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(xs, mix_y)
    plt.title("Mix Gaussian")
    plt.subplot(1, 2, 2)
    plt.hist(np.array(nums), bins = 200, normed = True)
    plt.title("M-H Sampling Mix Gaussian")
    plt.show()
    
def main():
    mh_sampling()
    
main()