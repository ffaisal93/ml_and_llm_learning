"""
Linear Algebra Q&A
Common interview questions with simple explanations
"""
import numpy as np

# ==================== Eigenvalues and Eigenvectors ====================

def eigenvalues_eigenvectors(A: np.ndarray) -> tuple:
    """
    Compute eigenvalues and eigenvectors
    
    Av = λv
    where A is matrix, v is eigenvector, λ is eigenvalue
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

def explain_eigenvalues():
    """
    Easy explanation:
    - Eigenvector: Direction that doesn't change when matrix is applied
    - Eigenvalue: How much it's scaled
    """
    print("Eigenvalues and Eigenvectors:")
    print("  - Eigenvector: Direction unchanged by matrix")
    print("  - Eigenvalue: Scaling factor")
    print("  - Formula: Av = λv")

# ==================== SVD ====================

def svd_decomposition(A: np.ndarray) -> tuple:
    """
    SVD: A = U Σ V^T
    
    U: Left singular vectors
    Σ: Singular values (diagonal)
    V: Right singular vectors
    """
    U, s, Vt = np.linalg.svd(A)
    return U, s, Vt

def explain_svd():
    """
    Easy explanation:
    - SVD decomposes any matrix into 3 parts
    - U: Left vectors (eigenvectors of AA^T)
    - Σ: Singular values (like eigenvalues)
    - V: Right vectors (eigenvectors of A^T A)
    """
    print("SVD (Singular Value Decomposition):")
    print("  - A = U Σ V^T")
    print("  - Works for any matrix (not just square)")
    print("  - Used in PCA, dimensionality reduction")

# ==================== Matrix Invertibility ====================

def is_invertible(A: np.ndarray) -> bool:
    """
    Check if matrix is invertible
    
    Matrix is invertible if:
    - Determinant ≠ 0
    - Full rank
    - All eigenvalues ≠ 0
    """
    det = np.linalg.det(A)
    return abs(det) > 1e-10

def explain_invertibility():
    """
    Easy explanation:
    - Invertible = has inverse matrix
    - Check: determinant ≠ 0
    - Or: full rank (no dependent rows/columns)
    """
    print("Matrix Invertibility:")
    print("  - Invertible if determinant ≠ 0")
    print("  - Or: full rank (linearly independent rows/columns)")
    print("  - A × A^(-1) = I (identity matrix)")

# ==================== Matrix Rank ====================

def matrix_rank(A: np.ndarray) -> int:
    """
    Compute matrix rank
    
    Rank = number of linearly independent rows/columns
    """
    return np.linalg.matrix_rank(A)

def explain_rank():
    """
    Easy explanation:
    - Rank = number of linearly independent rows/columns
    - Maximum number of independent vectors
    - Dimension of column space
    """
    print("Matrix Rank:")
    print("  - Number of linearly independent rows/columns")
    print("  - Maximum independent vectors")
    print("  - Dimension of column space")

# ==================== Positive Definite ====================

def is_positive_definite(A: np.ndarray) -> bool:
    """
    Check if matrix is positive definite
    
    Positive definite if:
    - All eigenvalues > 0
    - x^T A x > 0 for all x ≠ 0
    """
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

def explain_positive_definite():
    """
    Easy explanation:
    - Positive definite: All eigenvalues > 0
    - Semi-definite: All eigenvalues ≥ 0
    - Property: x^T A x > 0 for all x ≠ 0
    """
    print("Positive Definite Matrix:")
    print("  - All eigenvalues > 0")
    print("  - x^T A x > 0 for all x ≠ 0")
    print("  - Used in optimization (Hessian)")

# ==================== Common Operations ====================

def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication: A @ B"""
    return A @ B

def matrix_transpose(A: np.ndarray) -> np.ndarray:
    """Transpose: A^T"""
    return A.T

def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Inverse: A^(-1)"""
    return np.linalg.inv(A)

# ==================== Usage ====================

if __name__ == "__main__":
    print("Linear Algebra Q&A")
    print("=" * 60)
    print()
    
    # Example matrix
    A = np.array([[2, 1],
                  [1, 2]])
    
    print("Matrix A:")
    print(A)
    print()
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigenvalues_eigenvectors(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)
    print()
    
    explain_eigenvalues()
    print()
    
    # SVD
    U, s, Vt = svd_decomposition(A)
    print("SVD:")
    print(f"  U shape: {U.shape}")
    print(f"  Singular values: {s}")
    print(f"  V^T shape: {Vt.shape}")
    print()
    
    explain_svd()
    print()
    
    # Invertibility
    is_inv = is_invertible(A)
    print(f"Matrix is invertible: {is_inv}")
    explain_invertibility()
    print()
    
    # Rank
    rank = matrix_rank(A)
    print(f"Matrix rank: {rank}")
    explain_rank()
    print()
    
    # Positive definite
    is_pd = is_positive_definite(A)
    print(f"Matrix is positive definite: {is_pd}")
    explain_positive_definite()

