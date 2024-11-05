import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

K = 1; L2 = 2; eps0 = 1
xspan_3 = [-L2, L2]

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

slope_rk45 = np.polyfit(np.log(step_rk45), np.log(TOL), 1)[0]
slope_rk23 = np.polyfit(np.log(step_rk23), np.log(TOL), 1)[0]
slope_radau = np.polyfit(np.log(step_radau), np.log(TOL), 1)[0]
slope_BDF = np.polyfit(np.log(step_BDF), np.log(TOL), 1)[0]

A9 = [float(slope_rk45), float(slope_rk23), float(slope_radau), float(slope_BDF)]
print(A9)