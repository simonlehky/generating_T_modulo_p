# ==============================================================================================
#  Project      : GENERATING T MODULO P
#  File         : density_plot.py
#  Description  : This script calculates and visualizes the density of the set U^{RS}_{N,p} 
#                 for various values of N (matrix dimension) and p (field size).
#  Authors      : Farnaz Adib Yaghmaie, Kristian Hengster-Movric, Simon Lehky
#  Copyright    : (c) 2026 Farnaz Adib Yaghmaie / Linköping University
#  Year         : 2026
#  Repository   : https://github.com/simonlehky/generating_T_modulo_p
#  License      : Linköping University, Czech Technical University in Prague
# ==============================================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", 
                            "text.usetex": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def calculate_density(N, p):
    """
    Calculates the density (success probability) delta_N of generating an invertible RS matrix T over F_p.
    :param N: Matrix dimension
    :param p: Field size
    :return: Density of U^{RS}_{N,p} and the density excluding permutations.
    """
    N, p = int(N), int(p)
    if N <= 1: return 1.0   # for N=1, the matrix is T=[1], which is always invertible and RS; density is 1

    exponent = N - 1
    
    # 1. CALCULATE THE NUMERATOR: Product_{i=0}^{N-2} (p^{N-1} - p^i)
    numerator = 1.0
    p_float, exponent_float = float(p), float(exponent)     # use float for large intermediate values
    
    for i in range(N - 1):
        term = p_float**exponent_float - p_float**i
        if term <= 0:
            # Should not happen for p >= 2
            return 0.0 
        numerator *= term

    # 2. CALCULATE THE DENOMINATOR: p^((N-1)^2)
    denominator_exponent = float(exponent ** 2)
    denominator = p_float ** denominator_exponent
    
    # 3. CALCULATE THE FINAL DENSITY
    if denominator == 0: return 0.0
    # Outputing Product_{i=0}^{N-2} (p^{N-1} - p^i)/p^((N-1)^2), Product_{i=0}^{N-2} (p^{N-1} - p^i)/p^((N-1)^2) - N!/p^(N(N-1))
    return (numerator / denominator), (numerator / denominator - math.factorial(N) / p_float**(N * exponent))


def calculate_trs_density(N, p):
    """
    Calculates the density delta^TRS_{N,p} as specified in the paper. 
    :param N: Matrix dimension
    :param p: Field size
    :return: Density of U^{TRS}_{N,p}
    """
    N, p = int(N), int(p)
    if N <= 1: return 1.0   # for N=1 matrix is T=[1], which is always invertible and RS; density is 1

    p_f, n_f = float(p), float(N)

    # 1. CALCULATE THE NUMERATOR: (p-1)^(N-1) * p^((N-1)*(N-2)/2)
    term2 = (p_f - 1)**(n_f - 1)
    term3 = p_f**((n_f - 1) * (n_f - 2) / 2)
    numerator = term2 * term3 -1

    # 2. CALCULATE THE DENOMINATOR: p^(N-1) * Product_{i=0}^{N-2} (p^{N-1} - p^i)
    # Note: This denominator is equivalent to |G_{N,p}^{RS}|
    product_term = 1.0
    exponent = n_f - 1
    for i in range(N - 1):
        product_term *= (p_f**exponent - p_f**i)
        
    denominator = (p_f**(n_f - 1)) * product_term -  math.factorial(N)

    if denominator == 0: return 0.0
    return (numerator / denominator)


def calculate_trs_card(N, p):
    """
    Calculates the cardinality of U^{RS}_{N,p} as specified in the paper. 
    :param N: Matrix dimension
    :param p: Field size
    :return: Cardinality of U^{RS}_{N,p}    
    """
    N, p = int(N), int(p)
    if N <= 1: return 1.0   # for N=1 matrix is T=[1], which is always invertible and RS; density is 1

    p_f, n_f = float(p), float(N)
    return ((p_f - 1)**(n_f - 1) * p_f**((n_f - 1) * (n_f - 2) / 2) - 1)


# --- PLOTTING SETUP ---
# Define the range for N (Matrix Dimension) and p (Field Size)
# We keep N small because the exponential growth makes the computation unstable or impossible quickly.
N_range = np.arange(2, 10, 1)   # N = 2, 3, 4, 5,...
P_range = np.array([2, 3, 5, 7, 11, 13, 17, 19])  # p: prime numbers

# Create a mesh grid for plotting
X_N, Y_P = np.meshgrid(N_range, P_range)
Z_Density = np.zeros(X_N.shape)
Z_Density_excl_perm = np.zeros(X_N.shape)
Z_TRS = np.zeros(X_N.shape)
car_TRS = np.zeros(X_N.shape)

# Calculate the density Z for every (N, p) pair
for i in range(len(P_range)):
    for j in range(len(N_range)):
        print(f"Calculating density for N={X_N[i, j]}, p={Y_P[i, j]}")
        N_val = X_N[i, j]
        P_val = Y_P[i, j]
        Z_Density[i, j], Z_Density_excl_perm[i, j] = calculate_density(N_val, P_val)
        Z_TRS[i, j] = calculate_trs_density(N_val, P_val)
        car_TRS[i, j] = calculate_trs_card(N_val, P_val)

# Flatten the arrays for scatter plot
X_flat = X_N.flatten()
Y_flat = Y_P.flatten()
Z_flat = Z_Density.flatten()
Z_flat_excl_perm = Z_Density_excl_perm.flatten()
Z_flat_trs = Z_TRS.flatten()
car_flat_trs = car_TRS.flatten()


# --- GENERATE PLOTS WITH PERMUTATION ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 3D Plot
# surf = ax.plot_surface(X_N, Y_P, Z_Density, cmap='viridis', alpha=0.7, edgecolor='none')
# scatter = ax.scatter(X_flat, Y_flat, Z_flat, c='black', s=20, marker='o', alpha=0.8) 
# ax.set_xlabel('Matrix Dimension (N)')
# ax.set_ylabel('Field Size (p)')
# ax.set_zlabel(r'Density $\delta_{N,p}$')
# ax.set_title(r'Density $\delta_{N,p}^{RS}$ over $F_p$')
# ax.set_xticks(N_range)
# ax.set_yticks(P_range)
# ax.set_zlim(0, 1.0)
# fig.colorbar(surf, shrink=0.5, aspect=5, label='Density Value')
# plt.show()

# # Plot 1: δ vs N (for different p values)
# fig2, ax1 = plt.subplots(figsize=(10, 6))
# for i, p_val in enumerate(P_range):
#     ax1.plot(N_range, Z_Density[i, :], marker='o', label=f'p = {p_val}', linewidth=2)

# ax1.set_xlabel('Matrix Dimension (N)', fontsize=12)
# ax1.set_ylabel(r'Density $\delta_{N,p}$', fontsize=12)
# ax1.set_title(r'Density $\delta_{N,p}$ vs Matrix Dimension (N)', fontsize=14)
# ax1.legend(loc='best')
# ax1.grid(True, alpha=0.3)
# ax1.set_ylim(0, 1.0)
# plt.tight_layout()
# plt.show()

# # Plot 2: δ vs p (for different N values)
# fig3, ax2 = plt.subplots(figsize=(10, 6))
# for j, n_val in enumerate(N_range):
#     ax2.plot(P_range, Z_Density[:, j], marker='o', label=f'N = {n_val}', linewidth=2)

# ax2.set_xlabel('Field Size (p)', fontsize=12)
# ax2.set_ylabel(r'Density $\delta_{N,p}$', fontsize=12)
# ax2.set_title(r'Density $\delta_{N,p}$ vs Field Size (p)', fontsize=14)
# ax2.set_xticks(P_range)
# ax2.legend(loc='best')
# ax2.grid(True, alpha=0.3)
# ax2.set_ylim(0, 1.0)
# plt.tight_layout()
# plt.show()


# --- GENERATE PLOTS FOR EXCLUDING PERMUTATION DATASET ---
# 3D Surface for Z_Density_excl_perm
fig_ex = plt.figure(figsize=(10, 8))
ax_ex = fig_ex.add_subplot(111, projection='3d')
surf_ex = ax_ex.plot_surface(X_N, Y_P, Z_Density_excl_perm, cmap='plasma', alpha=0.8, edgecolor='none')
# Small black dots on the excl-perm surface
Zf_ex = Z_Density_excl_perm.flatten()
scatter_ex = ax_ex.scatter(X_flat, Y_flat, Zf_ex, c='black', s=20, marker='o', alpha=0.9)
ax_ex.set_title(r'Density $\delta_{N,p}^{RS}$ over $F_p$', fontsize=18)
ax_ex.set_xlabel(r'Matrix Dimension $N$', fontsize=18, labelpad=12)
plt.xticks(fontsize=16)
ax_ex.set_ylabel(r'Field Size $p$', fontsize=18, labelpad=12)
plt.yticks(fontsize=16)
ax_ex.set_zlabel(r'Density $\delta_{N,p}^{RS}$', fontsize=18, labelpad=12)
ax_ex.set_zlim(0, 1.0)
ax_ex.tick_params(axis='z', labelsize=16)
cbar = fig_ex.colorbar(surf_ex, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label('Density Value (excl. perm.)', fontsize=16)
cbar.ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.show()

# 2D Plot A: δ_excl vs N (for different p values)
fig4, ax4 = plt.subplots(figsize=(10, 6))
for i, p_val in enumerate(P_range):
    ax4.plot(N_range, Z_Density_excl_perm[i, :], marker='o', label=f'p = {p_val}', linewidth=2)
ax4.set_title(r'Density $\delta_{N,p}^{RS}$ vs Matrix Dimension', fontsize=18)
ax4.set_xlabel(r'Matrix Dimension $N$', fontsize=18)
plt.xticks(fontsize=16)
ax4.set_ylabel(r'Density $\delta_{N,p}^{RS}$', fontsize=18)
plt.yticks(fontsize=16)
ax4.set_ylim(0, 1.0)
ax4.legend(loc='best', fontsize=16)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2D Plot B: δ_excl vs p (for different N values)
fig5, ax5 = plt.subplots(figsize=(10, 6))
for j, n_val in enumerate(N_range):
    ax5.plot(P_range, Z_Density_excl_perm[:, j], marker='o', label=f'N = {n_val}', linewidth=2)
ax5.set_title(r'Density $\delta_{N,p}^{RS}$ vs Field Size', fontsize=18)
ax5.set_xlabel(r'Field Size $p$', fontsize=18)
plt.xticks(fontsize=16)
ax5.set_ylabel(r'Density $\delta_{N,p}^{RS}$', fontsize=18)
plt.yticks(fontsize=16)
ax5.set_ylim(0, 1.0)
ax5.legend(loc='best', fontsize=16)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- GENERATE TRS PLOTS (δ_{N,p}^{TRS}) ---
# 3D Surface for Z_TRS
fig_trs = plt.figure(figsize=(10, 8))
ax_trs = fig_trs.add_subplot(111, projection='3d')
surf_trs = ax_trs.plot_surface(X_N, Y_P, Z_TRS, cmap='cividis', alpha=0.8, edgecolor='none')
# Small white dots on TRS surface
Zf_trs = Z_TRS.flatten()
scatter_trs = ax_trs.scatter(X_flat, Y_flat, Zf_trs, c='white', s=18, marker='o', alpha=0.9)
ax_trs.set_title(r'Density $\delta_{N,p}^{\mathrm{TRS}}$ over $F_p$', fontsize=18)
ax_trs.set_xlabel(r'Matrix Dimension $N$', fontsize=18, labelpad=12)
plt.xticks(fontsize=16)
ax_trs.set_ylabel(r'Field Size $p$', fontsize=18, labelpad=12)
plt.yticks(fontsize=16)
ax_trs.set_zlabel(r'Density $\delta_{N,p}^{\mathrm{TRS}}$', fontsize=18, labelpad=12)
ax_trs.set_zlim(0, 1.0)
ax_trs.tick_params(axis='z', labelsize=16)
cbar = fig_trs.colorbar(surf_trs, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label('TRS Density', fontsize=16)
cbar.ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.show()

# 2D Plot A: δ_TRS vs N (for different p values)
fig6, ax6 = plt.subplots(figsize=(10, 6))
for i, p_val in enumerate(P_range):
    ax6.plot(N_range, Z_TRS[i, :], marker='o', label=f'p = {p_val}', linewidth=2)
ax6.set_title(r'Density $\delta_{N,p}^{\mathrm{TRS}}$ vs Matrix Dimension', fontsize=18)
ax6.set_xlabel(r'Matrix Dimension $N$', fontsize=18)
plt.xticks(fontsize=16)
ax6.set_ylabel(r'Density $\delta_{N,p}^{\mathrm{TRS}}$', fontsize=18)
plt.yticks(fontsize=16)
ax6.set_ylim(0, 0.30)
ax6.legend(loc='best', fontsize=16)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2D Plot B: δ_TRS vs p (for different N values)
fig7, ax7 = plt.subplots(figsize=(10, 6))
for j, n_val in enumerate(N_range):
    ax7.plot(P_range, Z_TRS[:, j], marker='o', label=f'N = {n_val}', linewidth=2)
ax7.set_title(r'Density $\delta_{N,p}^{\mathrm{TRS}}$ vs Field Size', fontsize=18)
ax7.set_xlabel(r'Field Size $p$', fontsize=18)
plt.xticks(fontsize=16)
ax7.set_ylabel(r'Density $\delta_{N,p}^{\mathrm{TRS}}$', fontsize=18)
plt.yticks(fontsize=16)
ax7.set_ylim(0, 0.30)
ax7.legend(loc='best', fontsize=16)
ax7.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3D Surface for Z_TRS
fig_car_trs = plt.figure(figsize=(10, 8))
ax_car_trs = fig_car_trs.add_subplot(111, projection='3d')
surf_car_trs = ax_car_trs.plot_surface(X_N, Y_P, car_TRS, cmap='cividis', alpha=0.8, edgecolor='none')
# Small white dots on TRS surface
cf_trs = car_TRS.flatten()
scatter_trs = ax_car_trs.scatter(X_flat, Y_flat, cf_trs, c='white', s=18, marker='o', alpha=0.9)
ax_car_trs.set_title(r'Cardinality of $U_{N,p}^{\mathrm{RS}}$ over $F_p$', fontsize=18)
ax_car_trs.set_xlabel(r'Matrix Dimension $N$', fontsize=18, labelpad=12)
plt.xticks(fontsize=16)
ax_car_trs.set_ylabel(r'Field Size $p$', fontsize=18, labelpad=12)
plt.yticks(fontsize=16)
ax_car_trs.set_zlabel(r'$|U_{N,p}^{\mathrm{RS}}|$', fontsize=18, labelpad=12)
ax_car_trs.tick_params(axis='z', labelsize=16)
cbar = fig_car_trs.colorbar(surf_car_trs, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label('Cardinality of TRS set', fontsize=16)
cbar.ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.show()
