import numpy as np
import matplotlib.pyplot as plt

L1 = 4
N = 81
xspan_1 = np.linspace(-L1, L1, 81)
dx = xspan_1[1] - xspan_1[0]
K = 1 

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
    return np.sqrt(norm_error)

def eigenvalue_error(numerical_eps, exact_eps):
    return 100 * np.abs(numerical_eps - exact_eps) / exact_eps

exact_eigenfunctions_matrix = np.zeros((N, 5))
for n in range(5):
    exact_eigenfunctions_matrix[:, n] = exact_eigenfunctions(n, xspan_1)
exact_eigenvalues_vector = np.array([exact_eigenvalues(n) for n in range(5)])

# eigenfunctions, eigenvalues to be replaced
eigenfunctions_shooting = np.random.random((N, 5)) 
eigenvalues_shooting = np.array([1, 3, 5, 7, 9])
eigenfunctions_direct = np.random.random((N, 5)) 
eigenvalues_direct = np.array([1, 3, 5, 7, 9]) 

eigenfunc_errors_shooting = np.zeros(5)
for n in range(5):
    eigenfunc_errors_shooting[n] = eigenfunction_error(eigenfunctions_shooting[:, n], exact_eigenfunctions_matrix[:, n])

eigenfunc_errors_direct = np.zeros(5)
for n in range(5):
    eigenfunc_errors_direct[n] = eigenfunction_error(eigenfunctions_direct[:, n], exact_eigenfunctions_matrix[:, n])

eigenvalue_errors_shooting = np.zeros(5)
for n in range(5):
    eigenvalue_errors_shooting[n] = eigenvalue_error(eigenvalues_shooting[n], exact_eigenvalues_vector[n])

eigenvalue_errors_direct = np.zeros(5)
for n in range(5):
    eigenvalue_errors_direct[n] = eigenvalue_error(eigenvalues_direct[n], exact_eigenvalues_vector[n])

print("Shooting Method Eigenfunction Errors:", eigenfunc_errors_shooting)
print("Shooting Method Eigenvalue Errors (%):", eigenvalue_errors_shooting)

print("Direct Method Eigenfunction Errors:", eigenfunc_errors_direct)
print("Direct Method Eigenvalue Errors (%):", eigenvalue_errors_direct)

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
