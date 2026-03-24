# ==============================================================================================
#  Project      : GENERATING T MODULO P
#  File         : triangular_form.py
#  Description  : This script implements the "Triangular-Form Method" for generating uniformly 
#                 random invertible row-stochastic matrices over finite fields F_p.
#  Authors      : Farnaz Adib Yaghmaie, Kristian Hengster-Movric, Simon Lehky
#  Copyright    : (c) 2026 Farnaz Adib Yaghmaie / Linköping University
#  Year         : 2026
#  Repository   : https://github.com/simonlehky/generating_T_modulo_p
#  License      : Linköping University, Czech Technical University in Prague
# ==============================================================================================

import numpy as np


def generate_triangular_row_stochastic_fixed_column(N, p, upper_triangular=True):
    """
    Generates an (N x N) matrix T over F_p that is nonsingular (invertible) and 
    row stochastic (RS), following the provided Triangular-Form algorithm.
    This algorithm uses either the last column (for UT) or the first column (for LT)
    as the dependent variable to enforce the row sum constraint.
    :param N: Dimension of the square matrix.
    :param p: Modulus (prime) for the finite field F_p.
    :param upper_triangular: If True, generates an Upper Triangular matrix.
                             If False, generates a Lower Triangular matrix.
    :return: An (N x N) matrix T that is row stochastic and invertible over F_p.
    :raises ValueError: If input parameters are invalid (e.g., N < 1 or p <= 1).
    """
    if N < 1:
        raise ValueError("Matrix dimension N must be 1 or greater.")
    if p <= 1:
        raise ValueError("Modulus p must be a prime number greater than 1.")
        
    T = np.zeros((N, N), dtype=np.int64)
    
    # 1. SAMPLE N-1 NONZERO DIAGONAL ELEMENTS
    # For N=1, this step is skipped (N-1 = 0)
    # Note: T[i, i] is sampled for i=0 to N-2 (N-1 elements).
    for i in range(N - 1):
        # We ensure T[i, i] is non-zero (p-1 choices) for nonsingularity
        T[i, i] = np.random.randint(1, p) 

    if upper_triangular:    # last column T[:, N-1] is dependent
        # 2. SAMPLE UPPER TRIANGULAR ELEMENTS EXCEPT THE LAST COLUMN
        # i runs from 0 to N-3 (Algorithm 1-based: i=1 to N-2)
        # j runs from i+1 to N-2 (Algorithm 1-based: j=i+1 to N-1)
        for i in range(N - 2):
            # Sample T[i, j] for j = i+1 to N-2
            T[i, i+1:N-1] = np.random.randint(0, p, N - 2 - i)

        # 3. SELECT THE LAST COLUMN TO BE RS
        # i runs from 0 to N-2 (Algorithm 1-based: i=1 to N-1)
        for i in range(N - 1):
            # Sum the predetermined parts of the row (T[i, i] up to T[i, N-2])
            # T[i, i:N-1] covers T[i, i], T[i, i+1],..., T[i, N-2]
            current_sum = np.sum(T[i, i:N-1]) % p
            # T[i, N-1] = 1 - current_sum (mod p)
            T[i, N-1] = (1 - current_sum) % p
            
            if T[i, N-1] < 0:
                T[i, N-1] += p
                
        # T[N, N] = 1 (Algorithm 1-based: T[N, N])
        T[N-1, N-1] = 1

    else:   # lower triangular case: first column T[:, 0] is dependent        
        # The algorithm implicitly fixes T[1, 1] = 1 (Algorithm 1-based: T[1, 1]=1)
        # and T[i, i] for i=2 to N-1 (Python: i=1 to N-1) were already sampled in Step 1.
        
        # 1. ENFORCE T[1, 1] = 1 CONSTRAINT (The T[1,1] pivot is fixed to 1 by RS)
        # Since T was *not* sampled in step 1, we fix it here and overwrite
        # the T[1,1] entry sampled earlier (i=1 in Python, i=2 in Algorithm 1 notation).
        T[0, 0] = 1 # T[1,1] = 1 in algorithm 1-based indexing

        # Step 1 specified sampling T[i,i] for i=1 to N-1. In Python, T is T_11.
        # We need to re-sample T[i,i] for i=1 to N-1 (N-1 to N in algo notation) 
        # that were originally sampled in the first N-1 elements.
        # Re-sample T[i, i] for i=1 to N-1 (T_22 to T_NN) to ensure (p-1) choices.
        for i in range(1, N):
            T[i, i] = np.random.randint(1, p) 
        
        # 2. SAMPLE LOWER TRIANGULAR ELEMENTS EXCEPT THE FIRST COLUMN
        # i runs from 2 to N-1 (Algorithm 1-based: i=3 to N)
        # j runs from 1 to i-1 (Algorithm 1-based: j=2 to i-1)
        for i in range(2, N):
            # Sample T[i, j] for j = 1 to i-1
            T[i, 1:i] = np.random.randint(0, p, i - 1)
            
        # 3. SELECT THE FIRST COLUMN TO BE RS
        # i runs from 1 to N-1 (Algorithm 1-based: i=2 to N)
        for i in range(1, N):
            # Sum the predetermined parts of the row (T[i, j] for j=2 up to T[i, i])
            # T[i, 1:i+1] covers T[i, 1], T[i, 2],..., T[i, i]
            current_sum = np.sum(T[i, 1:i+1]) % p
            
            # T[i, 0] = 1 - current_sum (mod p)
            T[i, 0] = (1 - current_sum) % p
            
            if T[i, 0] < 0:
                T[i, 0] += p
            
    # Final check for N=1 case: T must be 1 for RS and Nonsingular
    if N == 1: T = 1
    
    return T


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    N_dim = 2   # matrix dimension
    P_mod = 3   # finite field F_5 (entries in {0, 1, 2})

    # 1. GENERATE UPPER TRIANGULAR (UT) MATRIX
    T_UT = generate_triangular_row_stochastic_fixed_column(N=N_dim, p=P_mod, upper_triangular=True)
    print(f"Generated {N_dim}x{N_dim} Upper Triangular RS Matrix over F_{P_mod}:")
    print(T_UT)

    # UT VERIFICATION
    row_sums_UT = np.sum(T_UT, axis=1) % P_mod
    diag_prod_UT = np.prod(np.diag(T_UT)) % P_mod
    print(f"Row Sums Check (mod {P_mod}): {np.all(row_sums_UT == 1)}")
    print(f"Nonsingular Check (det={diag_prod_UT}!=0): {diag_prod_UT!= 0}\n")
    
    # 2. GENERATE LOWER TRIANGULAR (LT) MATRIX
    T_LT = generate_triangular_row_stochastic_fixed_column(N=N_dim, p=P_mod, upper_triangular=False)
    print(f"Generated {N_dim}x{N_dim} Lower Triangular RS Matrix over F_{P_mod}:")
    print(T_LT)

    # LT VERIFICATION
    row_sums_LT = np.sum(T_LT, axis=1) % P_mod
    diag_prod_LT = np.prod(np.diag(T_LT)) % P_mod
    print(f"Row Sums Check (mod {P_mod}): {np.all(row_sums_LT == 1)}")
    print(f"Nonsingular Check (det={diag_prod_LT}!=0): {diag_prod_LT!= 0}")
