import numpy as np
import matplotlib.pyplot as plt

lamd = 1.0             
shape = (5000, )       
bins = 200             


def inverse_cdf(lamd, shape):
    uni_nums = np.random.uniform(0.0, 1.0, shape)
    uni_nums[uni_nums == 1.0] = 0.99999
    exp_nums = -np.log(1 - uni_nums)/lamd
    return exp_nums


def sampling_exp():
    np_exp_nums = np.random.exponential(scale = 1.0/lamd, size = shape)   
    exp_nums = inverse_cdf(lamd = lamd, shape = shape)    
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(np_exp_nums, bins = bins, normed = True)
    plt.title("numpy exponential")
    plt.subplot(1, 2, 2)
    plt.hist(exp_nums, bins = bins, normed = True)
    plt.title("my exponential")
    plt.show()


def main():
    sampling_exp()


main()
