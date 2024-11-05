import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

tol = 1e-4
K = 1
L1 = 4
xspan_1 = np.linspace(-L1, L1, 81)
N = len(xspan_1) - 2
dx = xspan_1[1] - xspan_1[0]

def construct_A(K, xspan, dx):
    T1 = np.zeros((N, N)) 
    for i in range(N):
        T1[i, i] = -2 - K * (xspan[i+1])**2 * dx**2    # main diag
        for j in range(N-1):
            T1[j, j+1] = 1
            T1[j+1, j] = 1
    T1 = T1
    T2 = np.zeros((N, N))
    T2[0, 0] = 4/3
    T2[0, 1] = -1/3
    T2[N-1, N-2] = -1/3
    T2[N-1, N-1] = 4/3
    A = -(T1 + T2)
    return A

def sol_eigen(K, xspan, tol):
    col = ['r', 'b', 'g', 'c', 'm', 'k']
    eigenvalues = []
    eigenfunctions = []
    A = construct_A(K, xspan, dx)
    # print(A)
    # Solve A * phi = eps * phi
    eps_list, eigvecs = eigs(A / dx**2, k = 5, which = 'SM')
    sorted_indices = np.argsort(np.abs(eps_list))
    Vsort = eps_list[sorted_indices]
    Fsort = eigvecs[:,sorted_indices]

    for modes in range(5):
        phi_n = Fsort[:, modes]
        phi_n = np.insert(phi_n, 0, 4/3*phi_n[0] - 1/3*phi_n[1])
        phi_n = np.insert(phi_n, -1, 4/3*phi_n[-1] - 1/3*phi_n[-2])
        # Normalize
        norm = np.trapezoid(phi_n**2, xspan)
        norm_phi_n = phi_n / np.sqrt(norm)
        eigenvalues.append(Vsort[modes].real)
        eigenfunctions.append(np.abs(norm_phi_n))
        plt.plot(xspan, abs(norm_phi_n), col[modes])

    plt.title('First 5 Eigenfunctions (Direct Method)')
    plt.grid(True)
    plt.show()
    return eigenvalues, np.vstack(eigenfunctions)

eigenvalues_d, eigenfunctions_d = sol_eigen(K, xspan_1, tol)

A3 = eigenfunctions_d.T
A4 = [float(ev) for ev in eigenvalues_d]

print(A3)
print(A4)

