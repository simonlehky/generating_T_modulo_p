# ==============================================================================================
#  Project      : GENERATING T MODULO P
#  File         : sampling_rejection.py
#  Description  : This script implements the "Rejection Sampling Method" for generating 
#                 uniformly random invertible row-stochastic matrices over finite fields F_p.
#  Authors      : Farnaz Adib Yaghmaie, Kristian Hengster-Movric, Simon Lehky
#  Copyright    : (c) 2026 Farnaz Adib Yaghmaie / Linköping University
#  Year         : 2026
#  Repository   : https://github.com/simonlehky/generating_T_modulo_p
#  License      : Linköping University, Czech Technical University in Prague
# ==============================================================================================

import numpy as np


# --- MODULAR ARITHMETIC HELPERS ---
def mod_inverse(a, p):
    """
    Computes the modular multiplicative inverse of a mod p.
    :param a: Integer to invert
    :param p: Modulus (prime)
    :return: Modular inverse of a mod p, using Fermat's Little Theorem.
    :raises ValueError: If input matrix is singular or a pivot is zero
    """
    if a == 0:
        raise ValueError("Inverse does not exist for 0 mod p")
    return pow(int(a), int(p - 2), int(p))


def mod_det_gauss(M, p):
    """
    Computes the determinant of a matrix M modulo p using Gaussian elimination.
    This check forms the core of the rejection step.
    :param M: Square matrix
    :param p: Modulus (prime)
    :return: Determinant of M mod p
    """
    N = M.shape[0]
    M_copy = M.astype(np.int64)     # Use a copy to avoid modifying the original matrix
    sign = 1
    
    # Gaussian elimination to upper triangular form
    for i in range(N):
        # 1. PIVOTING: FIND THE FIRST NON-ZERO ELEMENT IN COLUMN i BELOW THE DIAGONAL
        pivot_row = i
        while pivot_row < N and M_copy[pivot_row, i] == 0:
            pivot_row += 1

        if pivot_row == N:  # If the entire column below is zero, the matrix is singular
            return 0 

        if pivot_row!= i:
            # Swap rows and update sign
            M_copy[[i, pivot_row]] = M_copy[[pivot_row, i]]
            sign *= -1

        # 2. ELIMINATION
        pivot = M_copy[i, i]
        
        # If we could not find a non-zero pivot, the matrix is singular (already handled above, but good safeguard)
        if pivot == 0: return 0 
            
        inv_pivot = mod_inverse(pivot, p)

        for j in range(i + 1, N):
            factor = (M_copy[j, i] * inv_pivot) % p
            
            # Apply elimination step: Row[j] = Row[j] - factor * Row[i] (mod p)
            M_copy[j, i:] = (M_copy[j, i:] - factor * M_copy[i, i:]) % p
            
            # Ensure all results are non-negative
            M_copy[j, i:] = M_copy[j, i:] % p

    # 3. DETERMINANT: PRODUCT OF THE DIAGONAL ENTRIES (MOD P)
    det_val = sign
    for i in range(N):
        det_val = (det_val * M_copy[i, i]) % p
        
    # Ensure final determinant is positive
    return (det_val % p)


# --- GENERATIVE MODEL 1: REJECTION SAMPLING ---
def generate_invertible_row_stochastic_rejection_sampling(N, p, max_attempts=10000):
    """
    Generates a uniformly random (N x N) matrix T over F_p that is both row stochastic and invertible 
    using Rejection Sampling.
    :param N: Dimension of the square matrix
    :param p: Modulus (prime)
    :param max_attempts: Maximum number of trials before giving up (to prevent infinite loops)
    :return: An (N x N) matrix T that is row stochastic and invertible over F_p
    :raises RuntimeError: If the method fails to generate a valid matrix after max_attempts
    """
    if p <= 1:
        raise ValueError("Modulus p must be a prime number greater than 1.")
        
    for attempt in range(max_attempts):
        # 1. GENERATE MATRIX T IN THE ROW STOCHASTIC AFFINE SUBSPACE M_{N,p}^{RS}
        T = np.zeros((N, N), dtype=np.int64)

        # We have N(N-1) free parameters to choose uniformly. [1]
        for i in range(N):
            # Sample N-1 entries uniformly from F_p
            # This samples the T[i, 0] through T[i, N-2] entries
            T[i, :N-1] = np.random.randint(0, p, N - 1, dtype=np.int64)

            # Calculate the N-th entry (T[i, N-1]) to ensure the row sum is 1 (mod p)
            # T[i, N-1] = 1 - sum(T[i, 0]...T[i, N-2]) mod p
            row_sum = np.sum(T[i, :N-1])
            T[i, N-1] = (1 - row_sum) % p
            
            # Ensure result is positive
            if T[i, N-1] < 0:
                T[i, N-1] += p

        # 2. CHECK INVERTIBILITY (REJECTION STEP)
        # We need the determinant to be non-zero mod p.[2]
        determinant = mod_det_gauss(T, p)

        # 3. ACCEPTANCE
        if determinant!= 0:
            # The matrix T is both RS and Invertible (in G_{N,p}^{RS})
            return T, attempt + 1
    
    # This should be rare since the success probability is bounded away from zero.
    raise RuntimeError(f"Failed to generate invertible RS matrix after {max_attempts} attempts. Check N and p. If p is large, consider increasing max_attempts.")


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Set parameters
    N_dim = 2  # matrix dimension
    P_mod = 3  # finite field F_5 (entries in {0, 1, 2, 3, 4})

    print(f"Generating {N_dim}x{N_dim} Invertible Row-Stochastic Matrix over F_{P_mod}...")

    try:
        T_matrix, attempts = generate_invertible_row_stochastic_rejection_sampling(N=N_dim, p=P_mod)

        print(f"\nSuccessfully generated matrix T in {attempts} attempt(s):")
        print(T_matrix)

        # Verification Checks (Mod P)
        print("\n--- Verification ---")
        
        # Check 1: Row Stochasticity (Row sum = 1 mod p)
        row_sums = np.sum(T_matrix, axis=1) % P_mod
        print(f"Row Sums (mod {P_mod}): {row_sums}")
        print(f"Row Stochastic Check: {np.all(row_sums == 1)}")

        # Check 2: Invertibility (Determinant!= 0 mod p)
        det_val = mod_det_gauss(T_matrix, P_mod)
        print(f"Determinant (mod {P_mod}): {det_val}")
        print(f"Invertibility Check: {det_val!= 0}")

    except RuntimeError as e:
        print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nError: {e}")
