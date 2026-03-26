# ==============================================================================================
#  Project      : GENERATING T MODULO P
#  File         : density_plot_refine.py
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
    Calculates the density of U^{RS}_{N,p} as specified in the paper.
    :param N: Matrix dimension
    :param p: Field size
    :return: Density of U^{RS}_{N,p} and the density excluding permutations.
    """
    N, p = int(N), int(p)
    if N <= 1: return 1.0, 1.0  # for N=1 matrix is T=[1], which is always invertible and RS; density is 1
    
    exponent = N - 1
    numerator = 1.0
    p_float = float(p)
    
    for i in range(N - 1):
        numerator *= (p_float**(float(exponent)) - p_float**i)

    denominator = p_float ** float(exponent ** 2)
    density = numerator / denominator
    excl_perm = density - math.factorial(N) / p_float**(N * exponent)
    return density, excl_perm


def calculate_trs_density(N, p):
    """
    Calculates the density of U^{TRS}_{N,p} as specified in the paper.
    :param N: Matrix dimension
    :param p: Field size
    :return: Density of U^{TRS}_{N,p}
    """
    N, p = int(N), int(p)
    if N <= 1: return 1.0   # for N=1 matrix is T=[1], which is always invertible and RS; density is 1
    p_f, n_f = float(p), float(N)
    
    numerator = (p_f - 1)**(n_f - 1) * p_f**((n_f - 1) * (n_f - 2) / 2) - 1
    
    product_term = 1.0
    for i in range(N - 1):
        product_term *= (p_f**(n_f - 1) - p_f**i)
        
    denominator = (p_f**(n_f - 1)) * product_term - math.factorial(N)
    return (numerator / denominator) if denominator != 0 else 0.0


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

    return (p_f - 1)**(n_f - 1) * p_f**((n_f - 1) * (n_f - 2) / 2) - 1


def plot_results(X, Y, Z, title, z_label, cmap='viridis', dot_color='black', is_density=True):
    """
    Refined plotting function: Generates one 3D surface and two 2D cross-sections.
    :param X: Meshgrid for N
    :param Y: Meshgrid for p
    :param Z: Density or cardinality values
    :param title: Title for the 3D plot
    :param z_label: Label for the Z-axis
    :param cmap: Colormap for the surface
    :param dot_color: Color for the scatter points
    :param is_density: If True, sets Z-axis limit to [0, 1]
    """
    # 1. 3D Surface Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.7, edgecolor='none')
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=dot_color, s=15, alpha=0.8)
    # ax.set_title(title, fontsize=18)
    ax.set_xlabel(r'Matrix dimension $N$', fontsize=18, labelpad=12)
    plt.xticks(fontsize=16)
    ax.set_ylabel(r'Field size $p$', fontsize=18, labelpad=12)
    plt.yticks(fontsize=16)
    ax.set_zlabel(z_label, fontsize=18, labelpad=12)
    ax.tick_params(axis='z', labelsize=16)
    if is_density: ax.set_zlim(0, 1.0)
    cbar = fig.colorbar(
    surf,
    orientation='horizontal',
    pad=0.05,        # distance below plot
    shrink=0.55,     # shorter bar
    fraction=0.05,   # thinner bar
    aspect=25
    )
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()

    # 2. 2D Plot: Z vs N (fixed p)
    plt.figure(figsize=(10, 5))
    for i, p_val in enumerate(Y[:, 0]):
        plt.plot(X[0, :], Z[i, :], marker='o', label=f'p = {p_val}')
    # plt.title(f"{z_label} vs Matrix Dimension", fontsize=20)
    plt.xlabel(r'Matrix dimension $N$', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel(z_label, fontsize=20)
    plt.yticks(fontsize=18)
    if is_density: plt.ylim(0, 1.1)
    plt.legend(loc='best', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. 2D Plot: Z vs p (fixed N)
    plt.figure(figsize=(10, 5))
    for j, n_val in enumerate(X[0, :]):
        plt.plot(Y[:, 0], Z[:, j], marker='s', label=f'N = {n_val}')
    # plt.title(f"{z_label} vs Field Size", fontsize=20)
    plt.xlabel(r'Field size $p$', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel(z_label, fontsize=20)
    plt.yticks(fontsize=18)
    if is_density: plt.ylim(0, 1.1)
    plt.legend(loc='best', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- PLOT DENSITY ---
N_range = np.arange(2, 10, 1)
P_range = np.array([2, 3, 5, 7, 11, 13, 17, 19])
X_N, Y_P = np.meshgrid(N_range, P_range)
Z_Density = np.zeros(X_N.shape)
Z_Density_excl = np.zeros(X_N.shape)
Z_TRS = np.zeros(X_N.shape)

for i in range(len(P_range)):
    for j in range(len(N_range)):
        Z_Density[i, j], Z_Density_excl[i, j] = calculate_density(X_N[i, j], Y_P[i, j])
        Z_TRS[i, j] = calculate_trs_density(X_N[i, j], Y_P[i, j])


# --- CALL THE UNIFIED PLOTTING FUNCTION ---
# plot_results(X_N, Y_P, Z_Density, r'Density $\delta_{N,p}^{RS}$ over $F_p$', r'Density $\delta$')
plot_results(X_N, Y_P, Z_Density_excl, r'Density', r'Density $\delta_{N,p}^{RS}$', cmap='plasma')
# plot_results(X_N, Y_P, Z_TRS, r'TRS Density $\delta_{N,p}^{TRS}$', r'TRS Density', cmap='cividis', dot_color='white')


# --- PLOT CARDINALITY ---
N_range = np.arange(2, 4, 1)        # N = 2, 3, 4, 5,...
P_range = np.array([2, 3, 5, 7])    # p: prime numbers
X_N, Y_P = np.meshgrid(N_range, P_range)
car_TRS = np.zeros(X_N.shape)

# Calculate the density Z for every (N, p) pair
for i in range(len(P_range)):
    for j in range(len(N_range)):
        print(f"Calculating cardinality for N={X_N[i, j]}, p={Y_P[i, j]}")
        car_TRS[i, j] = calculate_trs_card(X_N[i, j], Y_P[i, j])

plot_results(X_N, Y_P, car_TRS, 'Cardinality of $U_{N,p}^{RS}$', r'Cardinality $|U_{N,p}^{RS}|$', is_density=False)
