import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs

tol = 1e-4
K = 1
L1 = 4; L2 = 2
xspan_1 = np.linspace(-L1, L1, 81)
N = len(xspan_1) - 2
N2 = len(xspan_1)
dx = xspan_1[1] - xspan_1[0]
gamma_1 = 0.05; gamma_2 = -0.05
eps_init = 0.1; eps0 = 1
xp = [-L1, L1]
xspan_2 = np.linspace(-L2, L2, 41)
N1 = len(xspan_2)
gamma1 = 10; gamma2 = -10
xspan_3 = [-L2, L2]


# HW3_a
def shoot0(x, psi, eps):
    dpsi_dx = [psi[1], (K * x**2 - eps) * psi[0]]
    return dpsi_dx

def find_eigenvalues_and_eigenfunctions0():
    col = ['r', 'b', 'g', 'c', 'm', 'k']
    eigenvalues = []
    eigenfunctions = []
    eps_start = eps_init
    for modes in range(5):
        eps = eps_start
        deps = eps_init
        for j in range(1000):
            y0 = [1, np.sqrt(L1**2 - eps)]
            sol = solve_ivp(shoot0, (-L1, L1), y0, args=(eps,), t_eval=xspan_1)
            y = sol.y.T
            bc = y[-1, 1] + np.sqrt(L1**2 - eps) * y[-1, 0]
            if abs(bc - 0) < tol:
                eigenvalues.append(eps)
                break

            if (-1) ** (modes) * bc > 0:
                eps += deps
            else:
                deps /= 2
                eps -= deps
        eps_start = eps + 0.1
        # normalize eigenfunctions
        norm = np.trapezoid(y[:, 0] ** 2, xspan_1)
        y[:, 0] = y[:, 0] / np.sqrt(norm)
        eigenfunctions.append(np.abs(y[:, 0]))
        plt.plot(xspan_1, abs(y[:, 0]), col[modes])
    plt.title('First 5 Eigenfunctions (Shooting Method)')
    plt.grid(True)
    plt.show()
    
    return eigenvalues, np.vstack(eigenfunctions)


# HW3_b
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


# HW3_c
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


# HW3_d
def schrodinger(x, psi0):
    return [psi0[1], (K * x**2 - eps0) * psi0[0]]

def solve_tol(tol, method):
    # φ(0) = 1, φ'(0) = sqrt(K*L^2 - 1)
    y0 = [1, np.sqrt(K * L2**2 - eps0)]
    sol = solve_ivp(schrodinger, xspan_3, y0, method=method, atol=tol, rtol=tol, dense_output=True)
    avg_step_size = np.mean(np.diff(sol.t))
    return sol, avg_step_size

TOL = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
step_rk45 = []
step_rk23 = []
step_radau = []
step_BDF = []

for i in TOL:
    step_size_rk45 = solve_tol(i, 'RK45')[1]  # ODE45
    step_size_rk23 = solve_tol(i, 'RK23')[1]  # ODE23
    step_rk45.append(step_size_rk45)
    step_rk23.append(step_size_rk23)

step_rk45 = np.array(step_rk45)
step_rk23 = np.array(step_rk23)

plt.loglog(step_rk45, TOL, label="RK45 - 5th order")
plt.loglog(step_rk23, TOL, label="RK23 - 3rd order")
plt.xlabel("Average Step Size")
plt.ylabel("Tolerance")
plt.title("Log-Log Plot:")
plt.grid(True)
plt.legend()
plt.show()


for i in TOL:
    step_size_radau = solve_tol(i, 'Radau')[1]  # ODE113
    step_size_BDF = solve_tol(i, 'BDF')[1]   # ODE15s
        
    step_radau.append(step_size_radau)
    step_BDF.append(step_size_BDF)
    
step_radau = np.array(step_radau)
step_BDF = np.array(step_BDF)

plt.loglog(step_radau, TOL, label="Radau - 8th order")
plt.loglog(step_BDF, TOL, label="BDF - Implicit")
plt.xlabel("Average Step Size")
plt.ylabel("Tolerance")
plt.title("Log-Log Plot:")
plt.grid(True)
plt.legend()
plt.show()


# HW3_e
def exact_eigenvalues(n):
    return (2 * n + 1)

def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def exact_eigenfunctions(n, x):
    h = [1, 2*x, 4*x**2-2, 8*x**3-12*x, 16*x**4-48*x**2+12]
    H = h[n] * np.exp(-x**2 / 2) / np.sqrt(factorial(n) * 2**n * np.pi**0.5)
    return H

def eigenfunction_error(numerical_phi, exact_phi):
    error = np.abs(np.abs(numerical_phi) - np.abs(exact_phi))
    norm_error = np.trapezoid(error**2, xspan_1)
    return norm_error

def eigenvalue_error(numerical_eps, exact_eps):
    return 100 * np.abs(numerical_eps - exact_eps) / exact_eps

exact_eigenfunctions_matrix = np.zeros((N2, 5))
for n in range(5):
    exact_eigenfunctions_matrix[:, n] = exact_eigenfunctions(n, xspan_1)
exact_eigenvalues_vector = np.array([exact_eigenvalues(n) for n in range(5)])


eigenvalues_s, eigenfunctions_s = find_eigenvalues_and_eigenfunctions0()
eigenvalues_d, eigenfunctions_d = sol_eigen(K, xspan_1, tol)

eigenfunctions_shooting = eigenfunctions_s.T
eigenvalues_shooting = eigenvalues_s
eigenfunctions_direct = eigenfunctions_d.T
eigenvalues_direct = eigenvalues_d

eigenfunc_errors_s = np.zeros(5)
for n in range(5):
    eigenfunc_errors_s[n] = eigenfunction_error(eigenfunctions_shooting[:, n], exact_eigenfunctions_matrix[:, n])

eigenfunc_errors_d = np.zeros(5)
for n in range(5):
    eigenfunc_errors_d[n] = eigenfunction_error(eigenfunctions_direct[:, n], exact_eigenfunctions_matrix[:, n])

eigenvalue_errors_s = np.zeros(5)
for n in range(5):
    eigenvalue_errors_s[n] = eigenvalue_error(eigenvalues_shooting[n], exact_eigenvalues_vector[n])

eigenvalue_errors_d = np.zeros(5)
for n in range(5):
    eigenvalue_errors_d[n] = eigenvalue_error(eigenvalues_direct[n], exact_eigenvalues_vector[n])

for n in range(5):
    plt.plot(xspan_1, eigenfunctions_shooting[:, n])
    plt.plot(xspan_1, exact_eigenfunctions_matrix[:, n])

plt.title('Comparison of Shooting Method and Exact Eigenfunctions')
plt.grid(True)
plt.show()

for n in range(5):
    plt.plot(xspan_1, eigenfunctions_direct[:, n])
    plt.plot(xspan_1, exact_eigenfunctions_matrix[:, n])

plt.title('Comparison of Direct Method and Exact Eigenfunctions')
plt.grid(True)
plt.show()

slope_rk45 = np.polyfit(np.log(step_rk45), np.log(TOL), 1)[0]
slope_rk23 = np.polyfit(np.log(step_rk23), np.log(TOL), 1)[0]
slope_radau = np.polyfit(np.log(step_radau), np.log(TOL), 1)[0]
slope_BDF = np.polyfit(np.log(step_BDF), np.log(TOL), 1)[0]

A1 = eigenfunctions_s.T
A2 = eigenvalues_s
A3 = eigenfunctions_d.T
A4 = [float(ev) for ev in eigenvalues_d]
A9 = [float(slope_rk45), float(slope_rk23), float(slope_radau), float(slope_BDF)]
A10 = eigenfunc_errors_s
A11 = eigenvalue_errors_s
A12 = eigenfunc_errors_d
A13 = eigenvalue_errors_d

print("______a________")
print('A1: ', A1)
print('A2: ', A2)
print("______b________")
print('A3: ', A3)
print('A4: ', A4)
print("______c________")
print('A5: ', A5)
print('A6: ', A6)
print('A7: ', A7)
print('A8: ', A8)
print("______d________")
print('A9: ', A9)
print("______e________")
print('A10: ', A10)
print('A11: ', A11)
print('A12: ', A12)
print('A13: ', A13)