import scipy
import numpy as np
import random

if __name__ == "__main__":
    mnist = scipy.io.loadmat("../mnist-original.mat")
    mnist_data = mnist["data"].T
    mnist_label = mnist["label"][0]
    list_of_labels = []
    num_data_points = 10000
    for i in range(0,10):
        list_of_labels.append(list(mnist_label).index(i))
    list_of_labels.append(mnist_label.shape[0])
    random_samples = np.ones((10,1000))
    for i in range(0,10):
        my_samples = np.random.choice(range(int(list_of_labels[i]),int(list_of_labels[i+1])),size=1000,replace=False)
        random_samples[i] = my_samples
    sample_cov_matrix = np.zeros((784,784))
    mean = np.zeros((1,784))
    for sample in random_samples:
        for number in sample:
            img = mnist_data[int(number)]
            mean+= img
            transpose = img.transpose()
            sample_cov_matrix+= np.dot(img,img.transpose())
    sample_cov_matrix=sample_cov_matrix/10000
    mean= mean/10000
    eigenvalues,eigenvectors = np.linalg.eig(sample_cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    print(eigenvectors.shape)
    #start with your upper bound, like 250 or 200 vectors
    W_matrix = eigenvectors[::,0:250]
    pruned_values = eigenvalues[0:250]
    # print(W_matrix.shape)
    alphas = np.ones((250,))
    sigma_squared = 0
    for i in range(250,784):
        sigma_squared+=eigenvalues[i]
    sigma_squared = sigma_squared/(784-250)
    M_matrix = np.dot(W_matrix.transpose(),W_matrix) + sigma_squared*np.identity(250)
    M_inverse = np.linalg.inv(M_matrix)
    Ezn = np.zeros((10000,250,1),dtype="float64")
    Eznznt = np.zeros((10000,250,250),dtype="float64")
    for i in range(0,1000): #repeat the steps 1000 times
        for i in range(0,1): #10000 data entries
            #E step
            Ezn[i] = np.dot(np.dot(M_inverse,W_matrix.transpose()),(mnist_data[i,::]-mean).transpose())
            Eznznt[i] = sigma_squared*M_inverse + np.dot(Ezn[0],Ezn[0].transpose())
            #Start of M step
        left_side = np.zeros((784,250),dtype="float64")
        right_side = np.zeros((250,250),dtype="float64")
        for i in range(0,num_data_points):
            left_side+= np.dot((mnist_data[i,::]-mean).transpose(),Ezn[i].transpose())
            right_side+= Eznznt[i] +sigma_squared*np.identity(250)*alphas[i]
        W_new = np.dot(left_side,np.linalg.inv(right_side))

        #sigma step

        intermediate_sum = np.linalg.norm(mnist_data[0]-mean)**2 -np.dot(np.dot(2*Ezn[0],W_new.transpose()),mnist_data[0]-mean)
    



