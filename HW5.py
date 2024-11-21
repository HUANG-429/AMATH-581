import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, diags, identity, lil_matrix
from scipy.sparse.linalg import bicgstab, gmres
from scipy.linalg import lu, solve_triangular
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
import time


L = 20
nx, ny = 64, 64
m = 64
n = m * m
nu = 0.001
Lx, Ly = 20, 20
N = nx * ny
x2 = np.linspace(-L/2, L/2, nx+1)
x1 = x2[:nx]
dx = x1[1] - x1[0]
tspan = np.arange(0, 4.5, 0.5)

# Define grid and initial vorticity
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)


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

matA[0, 0] = 2 / (dx ** 2)
# print(matA)

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

A = matA
B = matB
C = matC

# initial condition
w0 = np.exp(-X**2 - Y**2 / 20)

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2


omega_flat = w0.reshape(N)
solve_times = {}

# FFT
def spc_rhs(t, wt0, nx, ny, N, K, nu):
    wtc = wt0.reshape((nx,ny))
    wt2 = fft2(wtc)
    psit_hat = -wt2 / K
    psit = np.real(ifft2(psit_hat)).reshape(N)
    rhs = nu * np.dot(A, wt0) - np.dot(B, psit) * np.dot(C, wt0) + np.dot(C, psit) * np.dot(B, wt0)
    return rhs

start_fft = time.time()
wtsol = solve_ivp(spc_rhs, (0, 4), omega_flat, t_eval=tspan, args=(nx, ny, N, K, nu))
end_fft = time.time()
solve_times['FFT'] = end_fft - start_fft
A1 = wtsol.y
print("A1:", A1)
print(np.shape(wtsol.y))

for j, t in enumerate(tspan):
    w = wtsol.y[:, j].reshape(nx, ny)
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'FFT Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()


# A \ b
def Ab_rhs(t, wt0, nu):
    psit = np.linalg.solve(A, wt0)
    rhs = nu * np.dot(A, wt0) - np.dot(B, psit) * np.dot(C, wt0) + np.dot(C, psit) * np.dot(B, wt0)
    return rhs
start_Ab = time.time()
psi_solivp = solve_ivp(Ab_rhs, (tspan[0], tspan[-1]), omega_flat, t_eval=tspan, args=(nu, ))
end_Ab = time.time()
solve_times['A/b'] = end_Ab - start_Ab
A2 = psi_solivp.y
print("A2:", A2)

for j, t in enumerate(tspan):
    w = psi_solivp.y[:, j].reshape(nx, ny)
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'A/b Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()


# LU decomposition
def lu_rhs(t, wt0, nu):
    P, L, U = lu(A)
    y = solve_triangular(L, np.dot(P, wt0), lower=True)
    psit = solve_triangular(U, y)
    rhs = nu * np.dot(A, wt0) - np.dot(B, psit) * np.dot(C, wt0) + np.dot(C, psit) * np.dot(B, wt0)
    return rhs
start_lu = time.time()
psi_lu = solve_ivp(lu_rhs, (tspan[0], tspan[-1]), omega_flat, t_eval=tspan, args=(nu, ))
end_lu = time.time()
solve_times['LU'] = end_lu - start_lu
A3 = psi_lu.y
print("A3:", A3)

for j, t in enumerate(tspan):
    w = psi_lu.y[:, j].reshape(nx, ny)
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'LU Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()


# Movie of FFT
wtsol_ym = wtsol.y.reshape(nx, ny, len(tspan)) 

fig, ax = plt.subplots()
cax = ax.pcolormesh(x, y, wtsol_ym[:, :, 0], cmap='viridis')
plt.colorbar(cax, ax=ax)
ax.set_title("Vorticity Evolution")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    cax.set_array(wtsol_ym[:, :, frame].flatten())
    ax.set_title(f"Time: {tspan[frame]:.2f}")
    return cax,

ani = FuncAnimation(fig, update, frames=len(tspan), interval=200, blit=True)
plt.show()

'''
# BICGSTAB
def Bicgstab_rhs(t, wt0, nu):
    psit, info_bicgstab = bicgstab(A, omega_flat, rtol=1e-2, maxiter=1000)
    rhs = nu * np.dot(A, wt0) - np.dot(B, psit) * np.dot(C, wt0) + np.dot(C, psit) * np.dot(B, wt0)
    return rhs
start = time.time()
psi_bicgstab = solve_ivp(Bicgstab_rhs, (tspan[0], tspan[-1]), omega_flat, t_eval=tspan, args=(nu, ))
solve_times['BICGSTAB'] = time.time() - start
print("BICGSTAB: ", psi_bicgstab.y)

for j, t in enumerate(tspan):
    w = psi_bicgstab.y[:, j].reshape(nx, ny)
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'BICGSTAB Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()


# GMRES
def Gmres_rhs(t, wt0, nu):
    psit, info_gmres = gmres(A, omega_flat, rtol=1e-2, maxiter=1000)
    rhs = nu * np.dot(A, wt0) - np.dot(B, psit) * np.dot(C, wt0) + np.dot(C, psit) * np.dot(B, wt0)
    return rhs
start = time.time()
psi_gmres = solve_ivp(Gmres_rhs, (tspan[0], tspan[-1]), omega_flat, t_eval=tspan, args=(nu, ))
solve_times['GMRES'] = time.time() - start
print("GMRES: ", psi_gmres.y)

for j, t in enumerate(tspan):
    w = psi_gmres.y[:, j].reshape(nx, ny)
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'GMRES Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()
'''

for method, time_taken in solve_times.items():
    print(f"{method} solve time: {time_taken:.4f} seconds")
