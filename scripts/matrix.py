import numpy as np 

M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(M.shape)
v = np.array([[1], 
             [2], 
             [3]])

print(M.dot(v))
# print(np.linalg.inv(M))
# print(np.linalg.det(M))
A = np.array([[0, 1],
             [-2, -3]])

eigen_values, eigen_vectors = np.linalg.eig(A)
print(eigen_values)
print(eigen_vectors)




