import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, diags, identity, lil_matrix

L = 20
m = 8  
n = m * m
x2 = np.linspace(-L/2, L/2, m+1)
x = x2[:m]
dx = x[1] - x[0]

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Matrix A
e0 = np.zeros((n, 1))  
e1 = np.ones((n, 1))   
e2 = np.copy(e1)       
e4 = np.copy(e0)       

# boundary conditions
for j in range(1, m + 1):
    e2[m * j - 1] = 0  # overwrite every m-th value with zero
    e4[m * j - 1] = 1  # overwrite every m-th value with one

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n - 1]
e3[0] = e2[n - 1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n - 1]
e5[0] = e4[n - 1]

diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(), 
               e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
               e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
matA = spdiags(diagonals_A, offsets_A, n, n).toarray() / (dx ** 2)


# Matrix B
I_m = identity(m)
O_m = lil_matrix((m, m))
B_upper_diag = I_m / (2 * dx)
B_lower_diag = -I_m / (2 * dx)

matB0 = lil_matrix((n, n))
# main diagonal blocks
for i in range(m):
    matB0[i*m:(i+1)*m, i*m:(i+1)*m] = O_m

# upper diagonal blocks
for i in range(m - 1):
    matB0[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = B_upper_diag

# lower diagonal blocks
for i in range(1, m):
    matB0[i*m:(i+1)*m, (i-1)*m:i*m] = B_lower_diag

# boundary condition blocks
matB0[:m, -m:] = B_lower_diag  # Top-right corner
matB0[-m:, :m] = B_upper_diag  # Bottom-left corner
matB = matB0.toarray()


# Matrix C
C_main_diag = np.zeros(m)
C_upper_diag = np.ones(m - 1)
C_lower_diag = -np.ones(m - 1)

I_0 = diags([C_main_diag, C_upper_diag, C_lower_diag], [0, 1, -1], shape=(m, m), format='lil')
# boundary conditions
I_0[0, m - 1] = -1  # Top-right corner
I_0[m - 1, 0] = 1   # Bottom-left corner

I_0_dense = I_0.toarray()
C = lil_matrix((n, n))

# main diagonal block
for i in range(m):
    C[i*m:(i+1)*m, i*m:(i+1)*m] = I_0 / (2 * dx)
matC = C.toarray()


A1 = matA
A2 = matB
A3 = matC

print(A1)
print(A2)
print(A3)
# print(I_0_dense)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.spy(A1)
plt.title('Matrix A (Laplacian)')

plt.subplot(1, 3, 2)
plt.spy(A2)
plt.title('Matrix B (∂_x)')

plt.subplot(1, 3, 3)
plt.spy(A3)
plt.title('Matrix C (∂_y)')

plt.show()
