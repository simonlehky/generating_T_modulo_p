# ==============================================================================================
#  Project      : GENERATING T MODULO P
#  File         : cardinality_plot.py
#  Description  : This script calculates and visualizes the cardinality of the set U^{RS}_{N,p} 
#                 for various values of N (matrix dimension) and p (field size).
#  Authors      : Farnaz Adib Yaghmaie, Kristian Hengster-Movric, Simon Lehky
#  Copyright    : (c) 2026 Farnaz Adib Yaghmaie / Linköping University
#  Year         : 2026
#  Repository   : https://github.com/simonlehky/generating_T_modulo_p
#  License      : Linköping University, Czech Technical University in Prague
# ==============================================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'text.usetex': False})


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
N_range = np.arange(2, 4, 1)    # N = 2, 3, 4, 5,...
P_range = np.array([2, 3, 5])   # p: prime numbers
X_N, Y_P = np.meshgrid(N_range, P_range)
car_TRS = np.zeros(X_N.shape)

# Calculate the density Z for every (N, p) pair
for i in range(len(P_range)):
    for j in range(len(N_range)):
        print(f"Calculating cardinality for N={X_N[i, j]}, p={Y_P[i, j]}")
        N_val = X_N[i, j]
        P_val = Y_P[i, j]
        car_TRS[i, j] = calculate_trs_card(N_val, P_val)

# Flatten the arrays for scatter plot
X_flat = X_N.flatten()
Y_flat = Y_P.flatten()
car_flat_trs = car_TRS.flatten()

# 3D Surface for Z_TRS
fig_car_trs = plt.figure(figsize=(10, 7))
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
