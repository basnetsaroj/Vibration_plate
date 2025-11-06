'''Problem: Rectangular plate's Free Vibration Analysis(mode shape, and frequency) with NACA 2412 Thickness Distribution
by Reissner Mindlin theory, Rayleigh Ritz method. [Clamped at x=0 and free at x=a, y=0,b (CFFF boundary condition)]''' 


import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ================================================================
#                       Plate parameters
# ================================================================
a = 0.15      # span (clamped at x=0, free at x=a)
b = 0.1       # chord (free at y=0 and y=b)
E = 200e9      
nu = 0.3
rho = 7850

n_bend = 6              # number of bending basis in x and y (for w)
n_tors = n_bend         # number of theta basis in x and y
n_mem = n_bend          # number of membrane basis in x and y (u,v)

N_bend = n_bend * n_bend
N_tors = n_tors * n_tors
N_mem = n_mem * n_mem

# total DOFs: w + theta_x + theta_y + u + v
total_dofs = N_bend + 2 * N_tors + 2 * N_mem

alpha = 5.0/6.0                 # or, np.pi**2 / 12.0   # shear correction factor 
G = E / (2.0 * (1.0 + nu))


# =================================================================================
#                       NACA Thickness Distribution
# =================================================================================
naca_code = "2412"
t = int(naca_code[2:]) / 100.0      # 12% for 2412

def naca_total_thickness(y):
    y = np.maximum(y, 1e-10)
    y_n = y/b                       # normalize by chord
    t_s = 5 * t * b * (0.2969 * np.sqrt(y_n) - 0.1260 * y_n- 0.3516 * y_n**2 + 0.2843 * y_n**3 - 0.1015 * y_n**4)
    return 2.0 * t_s                # total thickness (both sides)

def H(y):
    return np.maximum(naca_total_thickness(y), 1e-5) # avoid zero thickness


# =================================================================================
#                       BEAM FUNCTION ROOT FINDING
# =================================================================================
def find_beam_roots(f, df, initial_guesses, tol=1e-12, max_iter=100):
    roots = []
    for guess in initial_guesses:
        x = guess
        for iteration in range(max_iter):
            fx = f(x)
            dfx = df(x)
            if abs(dfx) < 1e-12:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                roots.append(x_new)
                break
            x = x_new
        else:
            roots.append(x)  
    return np.array(sorted(roots))

# Characteristic equations
f_cf = lambda L: np.cos(L) * np.cosh(L) + 1.0
df_cf = lambda L: -np.sin(L) * np.cosh(L) + np.cos(L) * np.sinh(L)
f_ff_even = lambda L: np.tan(L/2.0) + np.tanh(L/2.0)
df_ff_even = lambda L: 0.5 * (1/np.cos(L/2.0)**2 + 1/np.cosh(L/2.0)**2)
f_ff_odd = lambda L: np.tan(L/2.0) - np.tanh(L/2.0)
df_ff_odd = lambda L: 0.5 * (1/np.cos(L/2.0)**2 - 1/np.cosh(L/2.0)**2)

# Initial guesses for roots, taken from graph.
cf_guesses = [1.8751, 4.69409, 7.85476, 10.99554, 14.13717, 17.27876, 20.42035, 23.56194, 26.70354, 29.84513, np.pi*21/2, np.pi*23/2, np.pi*25/2, np.pi*27/2, np.pi*29/2, np.pi*31/2, np.pi*33/2, np.pi*35/2, np.pi*37/2, np.pi*39/2, np.pi*41/2]
ff_even_guesses = [4.730, 10.99561, 17.27876, 23.56194, 29.84513, np.pi*23/2, np.pi*27/2, np.pi*31/2, np.pi*35/2, np.pi*39/2]
ff_odd_guesses = [7.8532, 14.13717, 20.42035, 26.70354, np.pi*21/2, np.pi*25/2, np.pi*29/2, np.pi*33/2, np.pi*35/2, np.pi*41/2]

gamma_cf = find_beam_roots(f_cf, df_cf, cf_guesses[:n_bend])
gamma_ff_even = find_beam_roots(f_ff_even, df_ff_even, ff_even_guesses[:n_bend])
gamma_ff_odd = find_beam_roots(f_ff_odd, df_ff_odd, ff_odd_guesses[:n_bend])

sigma_cf = np.array([(np.sin(g) - np.sinh(g)) / (np.cos(g) + np.cosh(g)) for g in gamma_cf])
sigma_ff_even = np.array([np.sin(g/2.0) / np.sinh(g/2.0) for g in gamma_ff_even])
sigma_ff_odd = np.array([np.cos(g/2.0) / np.cosh(g/2.0) for g in gamma_ff_odd])


# =================================================================================
#                       BEAM EIGENFUNCTIONS
# =================================================================================
def X_clamped_free(x, i):
    g = gamma_cf[i]
    return (np.cos(g*x/a) - np.cosh(g*x/a)) + sigma_cf[i] * (np.sin(g*x/a) - np.sinh(g*x/a))

def dX_clamped_free_dx(x, i):
    g = gamma_cf[i]
    return (-g/a) * (np.sin(g*x/a) + np.sinh(g*x/a)) + sigma_cf[i] * (g/a) * (np.cos(g*x/a) - np.cosh(g*x/a))

def Y_free_free(y, m):
    if m == 0:
        return 1.0
    elif m == 1:
        return (1.0 - 2.0 * y / b)
    elif m % 2 == 0 and (m//2 - 1) < len(gamma_ff_even):
        idx = m//2 - 1
        g = gamma_ff_even[idx]
        s = sigma_ff_even[idx]
        return np.cos(g*(y/b - 0.5)) + s * np.cosh(g*(y/b - 0.5))
    elif m % 2 != 0 and (m//2 - 1) < len(gamma_ff_odd):
        idx = m//2 - 1
        g = gamma_ff_odd[idx]
        s = sigma_ff_odd[idx]
        return np.sin(g*(y/b - 0.5)) + s * np.sinh(g*(y/b - 0.5))
    else:
        return 0.0

def dY_free_free_dy(y, m):
    if m == 0:
        return 0.0
    elif m == 1:
        return -2.0 / b
    elif m % 2 == 0 and (m//2 - 1) < len(gamma_ff_even):
        idx = m//2 - 1
        g = gamma_ff_even[idx]
        s = sigma_ff_even[idx]
        return (-g/b) * np.sin(g*(y/b - 0.5)) + (g/b) * s * np.sinh(g*(y/b - 0.5))
    elif m % 2 != 0 and (m//2 - 1) < len(gamma_ff_odd):
        idx = m//2 - 1
        g = gamma_ff_odd[idx]
        s = sigma_ff_odd[idx]
        return (g/b) * np.cos(g*(y/b - 0.5)) + (g/b) * s * np.cosh(g*(y/b - 0.5))
    else:
        return 0.0


# =================================================================================
#                       BASIS FUNCTIONS FOR EACH FIELD
# =================================================================================
# w-field
def X_w(x, i): return X_clamped_free(x, i)
def Y_w(y, j): return Y_free_free(y, j)
def dX_w_dx(x, i): return dX_clamped_free_dx(x, i)
def dY_w_dy(y, j): return dY_free_free_dy(y, j)

# theta_x
def X_theta_x(x, i): return X_clamped_free(x, i)
def Y_theta_x(y, j): return Y_free_free(y, j)
def dX_theta_x_dx(x, i): return dX_clamped_free_dx(x, i)
def dY_theta_x_dy(y, j): return dY_free_free_dy(y, j)

# theta_y
def X_theta_y(x, i): return X_clamped_free(x, i)
def Y_theta_y(y, j): return Y_free_free(y, j)
def dX_theta_y_dx(x, i): return dX_clamped_free_dx(x, i)
def dY_theta_y_dy(y, j): return dY_free_free_dy(y, j)

# u (in-plane x)
def X_u(x, i): return X_clamped_free(x, i)
def Y_u(y, j): return Y_free_free(y, j)
def dX_u_dx(x, i): return dX_clamped_free_dx(x, i)
def dY_u_dy(y, j): return dY_free_free_dy(y, j)

# v (in-plane y)
def X_v(x, i): return X_clamped_free(x, i)
def Y_v(y, j): return Y_free_free(y, j)
def dX_v_dx(x, i): return dX_clamped_free_dx(x, i)
def dY_v_dy(y, j): return dY_free_free_dy(y, j)


# ================================================================
#                       Energy Integrands
# ================================================================
def integrand_stiffness_components(x, y, field1, X1_f, Y1_f, dX1dx_f, dY1dy_f, i1, j1, field2, X2_f, Y2_f, dX2dx_f, dY2dy_f, i2, j2):

    h = H(y)
    D_val = E * h**3 / (12.0 * (1.0 - nu**2))
    kGh = alpha * G * h

    X1, Y1 = X1_f(x, i1), Y1_f(y, j1)
    dX1_dx, dY1_dy = dX1dx_f(x, i1), dY1dy_f(y, j1)
    X2, Y2 = X2_f(x, i2), Y2_f(y, j2)
    dX2_dx, dY2_dy = dX2dx_f(x, i2), dY2dy_f(y, j2)

    # bending curvatures from theta fields
    kx1 = dX1_dx * Y1 if field1 == 'theta_x' else 0.0
    ky1 = X1 * dY1_dy if field1 == 'theta_y' else 0.0
    kxy1 = (X1 * dY1_dy if field1 == 'theta_x' else 0.0) + (dX1_dx * Y1 if field1 == 'theta_y' else 0.0)

    kx2 = dX2_dx * Y2 if field2 == 'theta_x' else 0.0
    ky2 = X2 * dY2_dy if field2 == 'theta_y' else 0.0
    kxy2 = (X2 * dY2_dy if field2 == 'theta_x' else 0.0) + (dX2_dx * Y2 if field2 == 'theta_y' else 0.0)

    bend_energy = D_val * (kx1*kx2 + ky1*ky2 + nu*(kx1*ky2 + kx2*ky1) + (1.0-nu)/2.0 * kxy1 * kxy2)

    # shear part (Mindlin: gamma = w_x - theta_x, gamma_y = w_y - theta_y)
    w_x1 = dX1_dx * Y1 if field1 == 'w' else 0.0
    w_y1 = X1 * dY1_dy if field1 == 'w' else 0.0
    theta_x1 = X1 * Y1 if field1 == 'theta_x' else 0.0
    theta_y1 = X1 * Y1 if field1 == 'theta_y' else 0.0

    w_x2 = dX2_dx * Y2 if field2 == 'w' else 0.0
    w_y2 = X2 * dY2_dy if field2 == 'w' else 0.0
    theta_x2 = X2 * Y2 if field2 == 'theta_x' else 0.0
    theta_y2 = X2 * Y2 if field2 == 'theta_y' else 0.0

    gamma_xz1 = w_x1 - theta_x1
    gamma_yz1 = w_y1 - theta_y1
    gamma_xz2 = w_x2 - theta_x2
    gamma_yz2 = w_y2 - theta_y2

    shear_energy = kGh * (gamma_xz1 * gamma_xz2 + gamma_yz1 * gamma_yz2)

    return bend_energy, shear_energy


def integrand_membrane(x, y, field1, X1_f, Y1_f, dX1dx_f, dY1dy_f, i1, j1, field2, X2_f, Y2_f, dX2dx_f, dY2dy_f, i2, j2):
    h = H(y)
    A_val = E * h / (1.0 - nu**2)
    A_mat = np.array([[1.0, nu, 0.0],
                      [nu, 1.0, 0.0],
                      [0.0, 0.0, (1.0 - nu)/2.0]]) * A_val

    X1, Y1 = X1_f(x, i1), Y1_f(y, j1)
    dX1dx, dY1dy = dX1dx_f(x, i1), dY1dy_f(y, j1)
    X2, Y2 = X2_f(x, i2), Y2_f(y, j2)
    dX2dx, dY2dy = dX2dx_f(x, i2), dY2dy_f(y, j2)

    # derivatives of u/v shape functions
    du1_dx = dX1dx * Y1
    du1_dy = X1 * dY1dy
    du2_dx = dX2dx * Y2
    du2_dy = X2 * dY2dy

    # build strain vectors depending on field choice
    # For 'u' basis: eps = [du_x/dx, 0, du_x/dy]  (εx, εy=0, γxy = du/dy + dv/dx)
    # For 'v' basis: eps = [0, dv/dy, dv/dx]
    if field1 == 'u':
        eps1 = np.array([du1_dx, 0.0, du1_dy])
    elif field1 == 'v':
        eps1 = np.array([0.0, du1_dy, du1_dx])  # careful: for v, du1_dx holds dv/dx when using X_v,Y_v naming
    else:
        eps1 = np.zeros(3)

    if field2 == 'u':
        eps2 = np.array([du2_dx, 0.0, du2_dy])
    elif field2 == 'v':
        eps2 = np.array([0.0, du2_dy, du2_dx])
    else:
        eps2 = np.zeros(3)

    return eps1 @ (A_mat @ eps2)

def integrand_mass(x, y, field1, X1_f, Y1_f, i1, j1, field2, X2_f, Y2_f, i2, j2):

    h = H(y)
    X1, Y1 = X1_f(x, i1), Y1_f(y, j1)
    X2, Y2 = X2_f(x, i2), Y2_f(y, j2)
    phi1 = X1 * Y1
    phi2 = X2 * Y2
    m = 0.0

    if field1 == 'w' and field2 == 'w':
        m += rho * h * phi1 * phi2

    rho_I = rho * h**3 / 12.0
    if field1 == 'theta_x' and field2 == 'theta_x':
        m += rho_I * phi1 * phi2
    if field1 == 'theta_y' and field2 == 'theta_y':
        m += rho_I * phi1 * phi2

    return m

def integrand_mass_membrane(x, y, field1, X1_f, Y1_f, i1, j1, field2, X2_f, Y2_f, i2, j2):

    h = H(y)
    X1, Y1 = X1_f(x, i1), Y1_f(y, j1)
    X2, Y2 = X2_f(x, i2), Y2_f(y, j2)
    phi1 = X1 * Y1
    phi2 = X2 * Y2
    m = 0.0
    if field1 == 'u' and field2 == 'u':
        m += rho * h * phi1 * phi2
    if field1 == 'v' and field2 == 'v':
        m += rho * h * phi1 * phi2
    return m


# ================================================================
#                       Matrix Assembly
# ================================================================
print("Assembling stiffness and mass matrices...")
K_bend = np.zeros((total_dofs, total_dofs))
K_shear = np.zeros((total_dofs, total_dofs))
K_mem = np.zeros((total_dofs, total_dofs))
K_total = np.zeros((total_dofs, total_dofs))
M = np.zeros((total_dofs, total_dofs))

# Gaussian quadrature
n_gauss = 16
x_points, x_weights = np.polynomial.legendre.leggauss(n_gauss)
y_points, y_weights = np.polynomial.legendre.leggauss(n_gauss)
x_scaled = 0.5 * a * (x_points + 1.0)
y_scaled = 0.5 * b * (y_points + 1.0)
x_weights_scaled = 0.5 * a * x_weights
y_weights_scaled = 0.5 * b * y_weights

# Build DOF map
dof_map = {}
for ij in range(total_dofs):
    if ij < N_bend:
        field = 'w'
        i, j = divmod(ij, n_bend)
        dof_map[ij] = (field, X_w, Y_w, dX_w_dx, dY_w_dy, i, j)
    elif ij < N_bend + N_tors:
        field = 'theta_x'
        i, j = divmod(ij - N_bend, n_tors)
        dof_map[ij] = (field, X_theta_x, Y_theta_x, dX_theta_x_dx, dY_theta_x_dy, i, j)
    elif ij < N_bend + 2 * N_tors:
        field = 'theta_y'
        i, j = divmod(ij - N_bend - N_tors, n_tors)
        dof_map[ij] = (field, X_theta_y, Y_theta_y, dX_theta_y_dx, dY_theta_y_dy, i, j)
    elif ij < N_bend + 2 * N_tors + N_mem:
        field = 'u'
        i, j = divmod(ij - N_bend - 2 * N_tors, n_mem)
        dof_map[ij] = (field, X_u, Y_u, dX_u_dx, dY_u_dy, i, j)
    else:
        field = 'v'
        i, j = divmod(ij - N_bend - 2 * N_tors - N_mem, n_mem)
        dof_map[ij] = (field, X_v, Y_v, dX_v_dx, dY_v_dy, i, j)

# double loop assemble
for ij in range(total_dofs):
    field1, X1_f, Y1_f, dX1dx_f, dY1dy_f, i1, j1 = dof_map[ij]
    for kl in range(ij, total_dofs):
        field2, X2_f, Y2_f, dX2dx_f, dY2dy_f, i2, j2 = dof_map[kl]
        integral_bend = 0.0
        integral_shear = 0.0
        integral_mem = 0.0
        integral_M = 0.0

        for y_idx, y_w in enumerate(y_weights_scaled):
            y = y_scaled[y_idx]
            for x_idx, x_w in enumerate(x_weights_scaled):
                x = x_scaled[x_idx]
                b_val, s_val = integrand_stiffness_components(
                    x, y,
                    field1, X1_f, Y1_f, dX1dx_f, dY1dy_f, i1, j1,
                    field2, X2_f, Y2_f, dX2dx_f, dY2dy_f, i2, j2
                )
                m_val = integrand_mass(
                    x, y,
                    field1, X1_f, Y1_f, i1, j1,
                    field2, X2_f, Y2_f, i2, j2
                )
                mem_val = integrand_membrane(
                    x, y,
                    field1, X1_f, Y1_f, dX1dx_f, dY1dy_f, i1, j1,
                    field2, X2_f, Y2_f, dX2dx_f, dY2dy_f, i2, j2
                )
                m_mem_val = integrand_mass_membrane(
                    x, y,
                    field1, X1_f, Y1_f, i1, j1,
                    field2, X2_f, Y2_f, i2, j2
                )

                wq = x_w * y_w
                integral_bend += b_val * wq
                integral_shear += s_val * wq
                integral_mem += mem_val * wq
                integral_M += (m_val + m_mem_val) * wq

        K_bend[ij, kl] = K_bend[kl, ij] = integral_bend
        K_shear[ij, kl] = K_shear[kl, ij] = integral_shear
        K_mem[ij, kl] = K_mem[kl, ij] = integral_mem
        K_total[ij, kl] = K_total[kl, ij] = integral_bend + integral_shear + integral_mem
        M[ij, kl] = M[kl, ij] = integral_M

    if (ij + 1) % 4 == 0 or ij == 0 or ij == total_dofs - 1:
        print(f"Completed row {ij + 1}/{total_dofs}")

# ================================================================
#                       Solve Eigenvalue Problem
# ================================================================
print("\nSolving eigenvalue problem...")
# Symmetrize
K_total = 0.5 * (K_total + K_total.T)
K_bend = 0.5 * (K_bend + K_bend.T)
K_shear = 0.5 * (K_shear + K_shear.T)
K_mem = 0.5 * (K_mem + K_mem.T)
M = 0.5 * (M + M.T)

# eigen
eigvals, eigvecs = eigh(K_total, M)
# filter
valid = eigvals > 1e-12
freq_hz = np.sqrt(eigvals[valid]) / (2.0 * np.pi)
eigvecs = eigvecs[:, valid]

# compute non-dimensional frequency parameter 
h_r = t*b    # reference thickness (max thickness)
Omega = a**2 * np.sqrt(12 * rho * (1 - nu**2) * eigvals[valid] / (E * h_r**2))  

# print frequencies
print("For n =", n_bend)
print("\nFirst 10 natural frequencies (Hz):")
for i in range(min(10, len(freq_hz))):
    print(f"{freq_hz[i]:7.2f}")

print("\nNon-dimensional frequency parameters (Ω):")
for i in range(min(10, len(Omega))):
    print(f"{Omega[i]:10.5f}")


# ================================================================
#                  Energy partition
# ================================================================
w_idx = np.arange(0, N_bend)
thx_idx = np.arange(N_bend, N_bend + N_tors)
thy_idx = np.arange(N_bend + N_tors, N_bend + 2 * N_tors)
u_idx = np.arange(N_bend + 2 * N_tors, N_bend + 2 * N_tors + N_mem)
v_idx = np.arange(N_bend + 2 * N_tors + N_mem, total_dofs)

print("\nMode    Eb%    Es%    Em% ")
for m in range(min(12, eigvecs.shape[1])):
    v = eigvecs[:, m]
    Etot = v.T @ K_total @ v
    Eb = v.T @ K_bend @ v
    Es = v.T @ K_shear @ v
    Em = v.T @ K_mem @ v
    def rms(indices):
        if len(indices) == 0: return 0.0
        return np.sqrt(np.mean(v[indices]**2))
    print(f"{m+1:4d} {Eb/Etot*100:5.1f}% {Es/Etot*100:5.1f}% {Em/Etot*100:5.1f}%")


def plot_3d_modes(num_modes=min(10, len(freq_hz))):
    # Create grid - flip X direction (clamped edge comes forward)
    y_vals = np.linspace(0.00001, b, 400)
    x_vals = np.linspace(a, 0, 400)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Create thickness profile
    Z_thickness = H(Y_grid) / 2
    Z_thickness[0, :] = 0        # Leading edge thickness = 0

    for mode in range(num_modes):
        # Initialize fields
        W = np.zeros_like(X_grid, dtype=float)
        U = np.zeros_like(X_grid, dtype=float)
        V = np.zeros_like(X_grid, dtype=float)

        # Reconstruct fields from eigenvector
        for ij in range(total_dofs):
            field, X_f, Y_f, _, _, i, j = dof_map[ij]
            coeff = np.real(eigvecs[ij, mode])
            Xval = X_f(X_grid, i)
            Yval = Y_f(Y_grid, j)
            if field == 'w':
                W += coeff * Xval * Yval
            elif field == 'u':
                U += coeff * Xval * Yval
            elif field == 'v':
                V += coeff * Xval * Yval
            # theta fields not used directly for geometry

        # Compute displacement magnitude and establish scaling
        disp_mag = np.sqrt(U**2 + V**2 + W**2)
        max_disp = np.max(np.abs(disp_mag))
        if max_disp == 0:
            scale = 1.0
        else:
            scale = 0.1 * max(a, b) / max_disp

        U_s = U * scale
        V_s = V * scale
        W_s = W * scale

        # Displaced coordinates
        X_disp = X_grid + U_s
        Y_disp = Y_grid + V_s
        Z_upper = Z_thickness + W_s
        Z_lower = -Z_thickness + W_s

        # Coloring: magnitude (or set color_data = W_s for signed)
        color_data = np.sqrt(U_s**2 + V_s**2 + W_s**2)
        max_c = np.max(np.abs(color_data))
        if max_c == 0: max_c = 1.0
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0.0, vmax=max_c)
        facecolors = cmap(norm(color_data))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Upper & lower surfaces
        ax.plot_surface(X_disp, Y_disp, Z_upper,
                        facecolors=facecolors, rstride=1, cstride=1,
                        antialiased=True, linewidth=0)
        ax.plot_surface(X_disp, Y_disp, Z_lower,
                        facecolors=facecolors, rstride=1, cstride=1,
                        antialiased=True, linewidth=0)

        # Side at x = first column (clamped edge)
        X_col0 = X_disp[:, 0]            # shape (Ny,)
        Y_col0 = Y_disp[:, 0]
        Zlow_col0 = Z_lower[:, 0]
        Zup_col0 = Z_upper[:, 0]
        # Make (Ny, 2) arrays for plot_surface
        X_side0 = np.column_stack([X_col0, X_col0])
        Y_side0 = np.column_stack([Y_col0, Y_col0])
        Z_side0 = np.column_stack([Zlow_col0, Zup_col0])
        colors_side0 = np.repeat(cmap(norm(color_data[:, 0]))[:, np.newaxis, :], 2, axis=1)  # (Ny,2,4)
        ax.plot_surface(X_side0, Y_side0, Z_side0, facecolors=colors_side0, rstride=1, cstride=1, linewidth=0)

        # Side at x = last column (free edge)
        X_col1 = X_disp[:, -1]
        Y_col1 = Y_disp[:, -1]
        Zlow_col1 = Z_lower[:, -1]
        Zup_col1 = Z_upper[:, -1]
        X_side1 = np.column_stack([X_col1, X_col1])
        Y_side1 = np.column_stack([Y_col1, Y_col1])
        Z_side1 = np.column_stack([Zlow_col1, Zup_col1])
        colors_side1 = np.repeat(cmap(norm(color_data[:, -1]))[:, np.newaxis, :], 2, axis=1)
        ax.plot_surface(X_side1, Y_side1, Z_side1, facecolors=colors_side1, rstride=1, cstride=1, linewidth=0)

        # Side at y = first row (y ~ 0.001)
        X_row0 = X_disp[0, :]
        Y_row0 = Y_disp[0, :]
        Zlow_row0 = Z_lower[0, :]
        Zup_row0 = Z_upper[0, :]
        X_side2 = np.vstack([X_row0, X_row0])
        Y_side2 = np.vstack([Y_row0, Y_row0])
        Z_side2 = np.vstack([Zlow_row0, Zup_row0])
        colors_side2 = np.repeat(cmap(norm(color_data[0, :]))[np.newaxis, :, :], 2, axis=0)
        ax.plot_surface(X_side2, Y_side2, Z_side2, facecolors=colors_side2, rstride=1, cstride=1, linewidth=0)

        # Side at y = last row (y = b)
        X_row1 = X_disp[-1, :]
        Y_row1 = Y_disp[-1, :]
        Zlow_row1 = Z_lower[-1, :]
        Zup_row1 = Z_upper[-1, :]
        X_side3 = np.vstack([X_row1, X_row1])
        Y_side3 = np.vstack([Y_row1, Y_row1])
        Z_side3 = np.vstack([Zlow_row1, Zup_row1])
        colors_side3 = np.repeat(cmap(norm(color_data[-1, :]))[np.newaxis, :, :], 2, axis=0)
        ax.plot_surface(X_side3, Y_side3, Z_side3, facecolors=colors_side3, rstride=1, cstride=1, linewidth=0)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(color_data)
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
        cbar.set_label('Displacement [m] (scaled)', fontsize=10)

        # Viewing and formatting
        ax.view_init(elev=30, azim=-125)
        ax.dist = 8  # zoom slightly closer

        ax.invert_yaxis()
        ax.zaxis.set_major_locator(MaxNLocator(4))  # at most 4 ticks on z-axis
        ax.set_xlabel('Span [m]', labelpad=12)
        ax.set_ylabel('Chord [m]', labelpad=12)
        ax.set_zlabel('Displacement [m]', labelpad=12)
        ax.set_title(f'\n\nMode {mode+1}. Frequency: {freq_hz[mode]:.1f} Hz', pad=20)
        ax.set_box_aspect([1, 1, 0.18])
        plt.tight_layout()
        plt.show()


plot_3d_modes(num_modes=min(10, len(freq_hz)))
