import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

tol = 1e-4
K = 1
L2 = 2
gamma_1 = 0.05
gamma_2 = -0.05
eps_init = 0.1
A_init = 0.1
xp = [-L2, L2]

xspan_2 = np.linspace(-L2, L2, 41)
N1 = len(xspan_2)

A6 = np.zeros(2) 
A5 = np.zeros((N1, 2)) 
A8 = np.zeros(2) 
A7 = np.zeros((N1, 2))

def shoot2(x, psi, eps, gamma):
    dpsi_dx = [psi[1], (gamma * np.abs(psi[0])**2 + K * x**2 - eps) * psi[0]]
    return dpsi_dx

gamma_values = [0.05, -0.05]
for gamma in gamma_values:
    E0 = 0.1 
    A = 1e-6
    col = ['r', 'b', 'g', 'c', 'm', 'k']
    for modes in range(2):
        dA = 0.01
        
        for j in range(100):
            eps = E0
            deps = 0.2 
            
            for _ in range(100):  
                y0 = [A, np.sqrt(L2**2 - eps) * A]
                sol = solve_ivp(shoot2, (-L2, L2), y0, args=(eps, gamma), t_eval=xspan_2)
                ys = sol.y.T  
                xs = sol.t
                bc = ys[-1, 1] + np.sqrt(L2**2 - eps) * ys[-1, 0]
                if abs(bc - 0) < tol:
                    break  
                
                if (-1) ** (modes) * bc > 0:
                    eps += deps
                else:
                    eps -= deps / 2
                    deps /= 2
                
            Area = np.trapezoid(ys[:, 0] ** 2, xs)
            if abs(Area - 1) < tol:
                   break  
            
            if Area < 1:
                A += dA
            else:
                A -= dA
                dA /= 2  
            
        if gamma > 0:
            A6[modes] = eps
            A5[:, modes] = np.abs(ys[:, 0])
        else:
            A8[modes] = eps
            A7[:, modes] = np.abs(ys[:, 0])
        
        E0 = eps + 0.2
        plt.plot(xspan_2, abs(ys[:, 0]), col[modes])
        plt.title('First 2 Eigenfunctions (Shooting Method)')
    plt.grid(True)
    plt.show()

print(A5)
print(A6)
print(A7)
print(A8)
print(len(xspan_2))
