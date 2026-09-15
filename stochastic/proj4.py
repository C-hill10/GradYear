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
    for sample in random_samples:
        for number in sample:
            img = mnist_data[int(number)]
            transpose = img.transpose()
            sample_cov_matrix+= img * img.transpose()
    sample_cov_matrix=sample_cov_matrix/10000
    eigenvalues,eigenvectors = np.linalg.eig(sample_cov_matrix)