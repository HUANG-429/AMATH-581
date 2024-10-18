import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 1e-6
K = 1; L = 4
eps_init = 0.1
xp = [-L, L]
xspan = np.linspace(-L, L, 81)

def shoot2(psi, x, eps):
    dpsi_dx = [psi[1], (K * x**2 - eps) * psi[0]]
    return dpsi_dx

def find_eigenvalues_and_eigenfunctions():
    col = ['r', 'b', 'g', 'c', 'm', 'k']
    eigenvalues = []
    eigenfunctions = []
    eps_start = eps_init
    for modes in range(5):
        eps = eps_start
        deps = eps_init
        for j in range(1000):
            x0 = [1, np.sqrt(L**2 - eps)]
            y = odeint(shoot2, x0, xspan, args = (eps,))
            bc = y[-1, 1] + np.sqrt(L**2 - eps) * y[-1, 0]
            if abs(bc - 0) < tol:
                eigenvalues.append(eps)
                break
                
            if (-1) ** (modes) * bc > 0:
                eps += deps
            else:
                deps /= 2
                eps -= deps
        eps_start = eps + 0.1
        #normalize eigenfunctions
        norm = np.trapezoid(y[:, 0] ** 2, xspan)
        y[:, 0] = y[:, 0] / np.sqrt(norm)
        eigenfunctions.append(np.abs(y[:, 0]))
        plt.plot(xspan, abs(y[:,0]), col[modes])
    plt.title('First 5 Eigenfunctions (Shooting Method)')
    plt.grid(True)
    plt.show()
    
    return eigenvalues, np.vstack(eigenfunctions)

eigenvalues, eigenfunctions = find_eigenvalues_and_eigenfunctions()

A1 = eigenfunctions.T
A2 = eigenvalues

'''
print(A1)
print(A2)
print(len(xspan))
'''

