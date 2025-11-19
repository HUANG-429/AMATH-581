import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
import time


nx, ny = 64, 64
N2 = 30
nu = 0.001
Lx, Ly = 20, 20
N = nx * ny
tspan = np.arange(0, 4.5, 0.5)

# Define grid and initial vorticity
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)


solve_times = {}

# FFT

# initial condition
m = 1      # number of spirals
beta = 1
D1 = 0.1
u = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

u_flat = fft2(u).reshape(N)
v_flat = fft2(v).reshape(N)

uv = np.hstack([u_flat, v_flat])
print(uv)
print(np.shape(uv))

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

def fft_rhs(t, uv, nx, ny, K):
    Ut = uv[:N].reshape((nx, ny))
    Vt = uv[N:].reshape((nx, ny))
    U = ifft2(Ut)
    V = ifft2(Vt)
    A_2 = U**2 + V**2
    lambda_A = 1 - A_2
    omega_A = -beta * A_2
    dU = lambda_A * U - omega_A * V
    dV = omega_A * U + lambda_A * V
    LU = -K * Ut
    LV = -K * Vt
    dU = (fft2(dU) + D1 * LU).reshape(N)
    dV = (fft2(dV) + D1 * LV).reshape(N)
    return np.hstack([dU, dV])

start_fft = time.time()
fftsol = solve_ivp(fft_rhs, (0, 4), uv, t_eval=tspan, args=(nx, ny, K), method='RK45')
end_fft = time.time()
solve_times['FFT'] = end_fft - start_fft
A1 = fftsol.y
print("A1:", A1)
print(np.shape(fftsol.y))

for j, t in enumerate(tspan):
    U = np.real(ifft2(fftsol.y[:N, j].reshape(nx, ny)))
    V = np.real(ifft2(fftsol.y[N:, j].reshape(nx, ny)))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, U+V)
    plt.title(f'FFT Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Chebychev

def cheb(N):
    if N == 0:
        D = 0.; x = 1.
    else:
        n = np.arange(0, N+1)
        x = np.cos(np.pi * n / N).reshape(N+1, 1)
        c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**n).reshape(N+1, 1)
        X = np.tile(x, (1, N+1))
        dX = X - X.T
        D = np.dot(c, 1./c.T) / (dX+np.eye(N+1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N+1)

D, x3 = cheb(N2)
print(np.shape(x3))
D[N2, :] = 0
D[0, :] = 0
D = 0.1 * D
D2 = np.dot(D, D)
xc = x3 * 10
yc = xc

XC, YC = np.meshgrid(xc, yc)
uc = np.tanh(np.sqrt(XC**2 + YC**2)) * np.cos(m * np.angle(XC + 1j*YC) - np.sqrt(XC**2 + YC**2))
vc = np.tanh(np.sqrt(XC**2 + YC**2)) * np.sin(m * np.angle(XC + 1j*YC) - np.sqrt(XC**2 + YC**2))

uc_flat = uc.flatten()
vc_flat = vc.flatten()

uv_c = np.hstack([uc_flat, vc_flat])

I = np.eye(len(D2))
La = np.kron(I, D2) + np.kron(D2, I)

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XC, YC, uc, cmap='gray')
plt.show()
'''

def cheb_rhs(t, uv, N2):
    U = uv[:((N2+1)**2)]
    V = uv[((N2+1)**2):]
    A_2 = U**2 + V**2
    lambda_A = 1 - A_2
    omega_A = -beta * A_2
    dU = lambda_A * U - omega_A * V + D1 * np.dot(La, U)
    dV = omega_A * U + lambda_A * V + D1 * np.dot(La, V)
    return np.hstack([dU.flatten(), dV.flatten()])

start_cheb = time.time()
chebsol = solve_ivp(cheb_rhs, (0, 4), uv_c, t_eval=tspan, args=(N2, ), method='RK45')
end_cheb = time.time()
solve_times['Chebychev'] = end_cheb - start_cheb
A2 = chebsol.y
print("A2:", A2)
print(np.shape(chebsol.y))

for j, t in enumerate(tspan):
    U2 = chebsol.y[:((N2+1)**2), j].reshape(N2+1, N2+1)
    # V2 = chebsol.y[((N2+1)**2):, j].reshape(N2+1, N2+1)
    UV = chebsol.y[:((N2+1)**2), j].reshape(N2+1, N2+1) + chebsol.y[((N2+1)**2):, j].reshape(N2+1, N2+1)
    plt.subplot(3, 3, j + 1)
    plt.pcolor(xc, yc, U2)
    plt.title(f'Cheb Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()

for method, time_taken in solve_times.items():
    print(f"{method} solve time: {time_taken:.4f} seconds")