import scipy
import numpy as np
import random

if __name__ == "__main__":
    mnist = scipy.io.loadmat("../mnist-original.mat")
    mnist_data = mnist["data"].T
    print(mnist_data.shape)
    mnist_label = mnist["label"][0]
    list_of_labels = []
    for i in range(0,10):
        list_of_labels.append(list(mnist_label).index(i))
    list_of_labels.append(mnist_label.shape[0])
    random_samples = np.ones((10,1000))
    for i in range(0,10):
        print(f'one row of samples is {random_samples[i].shape}')
        my_samples = np.random.choice(range(int(list_of_labels[i]),int(list_of_labels[i+1])),size=1000,replace=False)
        print(my_samples.shape)
        random_samples[i] = my_samples
    print(random_samples.shape)
    sample_cov_matrix = np.zeros((784,784))
    mean = np.zeros((1,784))
    for sample in random_samples:
        for number in sample:
            img = mnist_data[int(number)]
            mean+= img
            transpose = img.transpose()
            sample_cov_matrix+= img * img.transpose()
    sample_cov_matrix=sample_cov_matrix/10000
    mean= mean/10000
    eigenvalues,eigenvectors = np.linalg.eig(sample_cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    #start with your upper bound, like 250 or 200 vectors
    pruned_vectors = eigenvectors[0:250,::]
    pruned_values = eigenvalues[0:250,::]
    alphas = np.ones((1,250))
    sigma_squared = 0
    for i in range(250,784):
        sigma_squared+=eigenvalues[i][i]
    sigma_squared/(784-250)