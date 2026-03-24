# ==============================================================================================
#  Project      : GENERATING T MODULO P
#  File         : invertible_block.py
#  Description  : This script implements the "Invertible Block Method" for generating uniformly 
#                 random invertible row-stochastic matrices over finite fields F_p.
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
    Computes modular multiplicative inverse of a mod p.
    :param a: Integer to invert
    :param p: Modulus (prime)
    :return: Modular inverse of a mod p, using Fermat's Little Theorem.
    :raises ValueError: If input matrix is singular or a pivot is zero
    """
    a = int(a % p)  # ensure 'a' is a positive integer representative
    if a == 0:
        raise ValueError("Inverse does not exist for 0 mod p") 
    return pow(a, p - 2, p)


def mod_matrix_mult(A, B, p):
    """
    Computes matrix product A @ B modulo p.
    :param A: Left matrix
    :param B: Right matrix
    :param p: Modulus (prime)
    :return: (A @ B) mod p
    """
    return np.remainder(A @ B, p)


def mod_det_gauss(M, p):
    """
    Computes the determinant of a matrix M modulo p using Gaussian elimination.
    :param M: Square matrix
    :param p: Modulus (prime)
    :return: Determinant of M mod p
    """
    N = M.shape[0]
    M_copy = M.astype(np.int64)     # use a copy to avoid modifying the original matrix
    sign = 1
    
    # Gaussian elimination to upper triangular form
    for i in range(N):
        pivot_row = i
        while pivot_row < N and M_copy[pivot_row, i] == 0:
            pivot_row += 1

        if pivot_row == N: return 0     # singular matrix

        if pivot_row!= i:
            M_copy[[i, pivot_row]] = M_copy[[pivot_row, i]]
            sign *= -1

        pivot = M_copy[i, i]
        inv_pivot = mod_inverse(pivot, p)

        for j in range(i + 1, N):
            if M_copy[j, i] == 0:
                continue
            factor = (M_copy[j, i] * inv_pivot) % p
            M_copy[j, i:] = np.remainder(M_copy[j, i:] - factor * M_copy[i, i:], p)

    det_val = sign
    for i in range(N):
        det_val = (det_val * M_copy[i, i]) % p
        
    return (det_val % p)


# --- INVERTIBLE BLOCK METHOD ---
def is_linearly_dependent_mod_p(basis_vectors, new_vector, p):
    """
    Checks if 'new_vector' is in the span of 'basis_vectors' using Gaussian elimination.
    The basis_vectors must be already linearly independent for this to correctly check
    if the rank increases by exactly 1 when adding new_vector.
    :param basis_vectors: Array of shape (M, k) containing M basis vectors of dimension k.
    :param new_vector: Array of shape (k,) representing the new vector to check.
    :param p: Modulus (prime)
    :return: True if 'new_vector' is linearly dependent on 'basis_vectors', False otherwise.
    """
    if basis_vectors.size == 0:     # if basis is empty, only the zero vector is dependent.
        return np.all(new_vector == 0)

    # 1. CREATE THE AUGMENTED MATRIX
    matrix = np.vstack([basis_vectors, new_vector]).astype(np.int64)
    M, k = matrix.shape     # M = number of vectors, k = dimension of vectors
    
    # 2. GAUSSIAN ELIMINATION TO FIND RANK
    rank = 0
    M_copy = np.copy(matrix)
    
    for j in range(k):  # iterate columns
        pivot_row = rank
        while pivot_row < M and M_copy[pivot_row, j] == 0:
            pivot_row += 1

        if pivot_row < M:
            # Found pivot: Perform row swap and elimination
            M_copy[[rank, pivot_row]] = M_copy[[pivot_row, rank]]
            
            pivot = M_copy[rank, j]
            inv_pivot = mod_inverse(pivot, p)
            
            for i in range(rank + 1, M):
                factor = (M_copy[i, j] * inv_pivot) % p
                M_copy[i, j:] = np.remainder(M_copy[i, j:] - factor * M_copy[rank, j:], p)
            rank += 1
            
    # The new vector is linearly dependent if the rank did not increase
    # Rank of basis_vectors is M-1 (since they are independently chosen).
    # If the new rank is still M-1, the new vector is dependent.
    return (rank == M - 1)


def generate_uniform_GL_constructive(k, p, max_attempts_per_row=1000):
    """
    Generates a uniformly random (k x k) matrix A_k over F_p that is invertible 
    by sequentially selecting rows linearly independent of the previous rows.
    :param k: Dimension of the square matrix
    :param p: Modulus (prime)
    :param max_attempts_per_row: Maximum attempts to find an independent row before giving up
    :return: A uniformly random matrix in GL_k(F_p)
    :raises RuntimeError: If unable to find an independent row after max attempts
    """
    if k == 0:
        return np.zeros((0, 0), dtype=np.int64)
        
    A_rows = []
    
    for i in range(k):      # generate the i-th row (i starts at 0)
        found_independent_row = False
        
        # We rely on the known non-zero success probability (bounded away from zero)
        for _ in range(max_attempts_per_row): 
            # 1. SAMPLE A RANDOM VECTOR UNIFORMLY
            new_row = np.random.randint(0, p, k, dtype=np.int64)
            
            # 2. CHECK LINEAR INDEPENDENCE
            current_basis = np.array(A_rows)
            
            if not is_linearly_dependent_mod_p(current_basis, new_row, p):
                A_rows.append(new_row)
                found_independent_row = True
                break
        
        if not found_independent_row:
             raise RuntimeError(f"Failed to find independent row {i+1} after {max_attempts_per_row} attempts.")

    return np.array(A_rows)


# --- GENERATIVE MODEL 2: TRANSFORMATIONAL CONSTRUCTIVE APPROACH (UPDATED) ---
def generate_invertible_row_stochastic_constructive(N, p):
    """
    Generates a uniformly random (N x N) matrix T over F_p that is both row stochastic and invertible,
    using the constructive T = P @ A @ P_inv method, with the GL submatrix generated constructively.
    :param N: Dimension of the square matrix
    :param p: Modulus (prime)
    :return: An (N x N) matrix T that is row stochastic and invertible over F_p
    :raises RuntimeError: If the GL generation fails after maximum attempts
    :raises ValueError: If input parameters are invalid (e.g., N < 1
    """
    if N < 1:
        raise ValueError("Matrix dimension N must be 1 or greater.")
    if p <= 1:
        raise ValueError("Modulus p must be a prime number greater than 1.")
        
    N_k = N - 1

    # 1. GENERATE A_{N-1} IN GL_{N-1}(F_p) UNIFORMLY AND CONSTRUCTIVELY
    # This step now relies on sequential row construction without a final determinant call.
    A_N_minus_1 = generate_uniform_GL_constructive(N_k, p)

    # 2. GENERATE c^T UNIFORMLY ((1 x N-1) ROW VECTOR)
    c_T = np.random.randint(0, p, N_k, dtype=np.int64)
    
    # 3. ASSEMBLE THE STABILIZED MATRIX A (N x N) GIVEN BY A = [A_{N-1} | 0]
    A = np.zeros((N, N), dtype=np.int64)
    if N_k > 0:
        A[:N_k, :N_k] = A_N_minus_1
        A[N_k, :N_k] = c_T
    A[N_k, N_k] = 1

    # 4. DEFINE P AND P_inv
    # P: I_{N} with the last column replaced by the all-ones vector 1.
    P = np.eye(N, dtype=np.int64)
    P[:, N_k] = 1 
    
    # P_inv: I_{N} with the last column entries (0 to N-2) replaced by p-1 (-1 mod p).
    P_inv = np.eye(N, dtype=np.int64)
    if N_k > 0:
        P_inv[:N_k, N_k] = p - 1 

    # 5. COMPUTE T = P @ A @ P_inv (mod p)
    PA = mod_matrix_mult(P, A, p)
    T = mod_matrix_mult(PA, P_inv, p)
    
    return T


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Set parameters
    N_dim = 2   # matrix dimension
    P_mod = 3   # finite field F_3 (entries in {0, 1, 2})

    print(f"Generating {N_dim}x{N_dim} Invertible Row-Stochastic Matrix over F_{P_mod} using Model 2 (Constructive GL generation)...")

    try:
        T_matrix = generate_invertible_row_stochastic_constructive(N=N_dim, p=P_mod)

        print("\nSuccessfully generated matrix T:")
        print(T_matrix)

        # Verification Checks (Mod P)
        print("\n--- Verification ---")
        
        # Check 1: Row Stochasticity (Row sum = 1 mod p)
        row_sums = np.sum(T_matrix, axis=1) % P_mod
        print(f"Row Sums (mod {P_mod}): {row_sums}")
        print(f"Row Stochastic Check (T@1=1): {np.all(row_sums == 1)}")

        # Check 2: Invertibility (Determinant!= 0 mod p)
        det_val = mod_det_gauss(T_matrix, P_mod)
        print(f"Determinant (mod {P_mod}): {det_val}")
        print(f"Invertibility Check (det!=0): {det_val!= 0}")

    except RuntimeError as e:
        print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nError: {e}")
