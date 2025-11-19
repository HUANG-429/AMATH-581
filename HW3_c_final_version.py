import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
L = 2
tol = 1e-4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)

# Save eigenvalues and the absolute values of eigenfunctions
A6 = np.zeros(2)  # For gamma = 0.05 eigenvalues
A5 = np.zeros((n, 2))  # For gamma = 0.05 eigenfunctions
A8 = np.zeros(2)  # For gamma = -0.05 eigenvalues
A7 = np.zeros((n, 2))  # For gamma = -0.05 eigenfunctions

# Define the right-hand side of the ODE
def bvp_rhs(x, y, epsilon, gamma):
    return [y[1], (gamma * y[0]**2 + x**2 - epsilon) * y[0]]

# Define gamma values to loop over
gamma_values = [0.05, -0.05]

# Searching for eigenvalues and eigenfunctions for gamma values 0.05 and -0.05
for gamma in [0.05, -0.05]:
    E0 = 0.1  # Initial guess for epsilon
    A = 1e-6  # Initial amplitude guess
    
    for modes in range(2):  # Searching for first two eigenvalues
        dA = 0.01  # Step size for adjusting amplitude
        
        for j in range(100):  # Outer loop for finding the eigenvalue
            epsilon = E0
            dep = 0.2  # Step size for epsilon adjustment
            
            for _ in range(100):  # Inner loop for epsilon adjustment
                # Initial condition with adjusted amplitude A
                y0 = [A, np.sqrt(L**2 - epsilon) * A]
                
                # Solve the BVP using solve_ivp
                sol = solve_ivp(lambda x, y: bvp_rhs(x, y, epsilon, gamma), [x[0], x[-1]], y0, t_eval=x)
                ys = sol.y.T  # Transposed for easier indexing
                xs = sol.t
                
                # Boundary condition at the endpoint
                if abs(ys[-1, 1] + np.sqrt(L**2 - epsilon) * ys[-1, 0]) < tol:
                    break  # Exit if boundary condition is satisfied
                
                # Adjust epsilon based on boundary condition sign
                if (-1) ** modes * (ys[-1, 1] + np.sqrt(L**2 - epsilon) * ys[-1, 0]) > 0:
                    epsilon += dep
                else:
                    epsilon -= dep / 2
                    dep /= 2  # Reduce step size for finer adjustment
                
            # Normalize the eigenfunction using area under |y|^2
            Area = np.trapezoid(ys[:, 0] ** 2, xs)
            if abs(Area - 1) < tol:
                break  # Exit if eigenfunction is normalized
            
            # Adjust A if the area deviates from 1
            if Area < 1:
                A += dA
            else:
                A -= dA
                dA /= 2  # Reduce step size for finer adjustment
            
        # Store results for the found eigenvalue and normalized eigenfunction
        if gamma > 0:
            A6[modes] = epsilon
            A5[:, modes] = np.abs(ys[:, 0])
        else:
            A8[modes] = epsilon
            A7[:, modes] = np.abs(ys[:, 0])
        
        # Update epsilon guess for next mode
        E0 = epsilon + 0.2

# Plot the results
plt.figure(figsize=(12, 6))
for i, (ysol, title) in enumerate([(A5, "gamma = 0.05"), (A7, "gamma = -0.05")]):
    plt.subplot(1, 2, i + 1)
    for j in range(2):
        plt.plot(x, ysol[:, j], label=f'|y_{j+1}| for mode {j+1}')
    plt.xlabel('x')
    plt.ylabel('|y| (Normalized)')
    plt.title(f'Normalized Eigenfunctions for {title}')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()

#print("Eigenvalues for gamma = 0.05:", A6)
#print("Eigenvalues for gamma = -0.05:", A8)


# Save the results
np.save('A6.npy', A6)  # Save eigenvalues for gamma = 0.05
np.save('A5.npy', A5)  # Save eigenfunctions for gamma = 0.05
#print('A5 =', A5)
print('A6 =', A6)
np.save('A8.npy', A8)  # Save eigenvalues for gamma = -0.05
np.save('A7.npy', A7)  # Save eigenfunctions for gamma = -0.05
#print('A7 =',A7)
print('A8 =', A8)