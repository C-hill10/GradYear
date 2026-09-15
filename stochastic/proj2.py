import numpy as np

if __name__ == "__main__":
    cov_matrix = np.matrix("4,2,2,2,2; 2,4,2,2,2; 2,2,4,2,2; 2,2,2,4,2; 2,2,2,2,4")
    values,vectors = np.linalg.eig(cov_matrix)
    sqrt_values = np.matrix([[values[0]**(1/2),0,0,0,0],[0,values[1]**(1/2),0,0,0],[0,0,values[2]**(1/2),0,0],[0,0,0,values[3]**(1/2),0],[0,0,0,0,values[4]**(1/2)]])
    t_matrix = vectors.transpose() * sqrt_values * vectors
    print(t_matrix)
    random_vectors = np.random.normal(0,1,(1000000,5,1))
    mean_vector = np.matrix([[1],[2],[3],[4],[5]])
    for rv in random_vectors:
        rv = t_matrix*rv + mean_vector