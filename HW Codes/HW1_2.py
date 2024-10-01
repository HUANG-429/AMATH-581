import numpy as np

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3],[0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

res_a = A + B
res_b = 3*x - 4*y
res_c = np.dot(A, x)
res_d = np.dot(B, (x-y))
res_e = np.dot(D, x)
res_f = np.dot(D, y) + z
res_g = np.dot(A, B)
res_h = np.dot(B, C)
res_i = np.dot(C, D)

np.save('A4.npy',res_a)
np.save('A5.npy',res_b)
np.save('A6.npy',res_c)
np.save('A7.npy',res_d)
np.save('A8.npy',res_e)
np.save('A9.npy',res_f)
np.save('A10.npy',res_g)
np.save('A11.npy',res_h)
np.save('A12.npy',res_i)
