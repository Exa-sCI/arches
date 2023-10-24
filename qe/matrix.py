import numpy as np
from scipy.linalg import lapack as la


def diagonalize(X):
    """Diagonalize matrix and return eigenbasis and corresponding eigenvalues

        diagonalize s.t. Z @ L @ Z.T = A
    Args:
        X (array) : (symmetric) matrix to be diagonalized

    Returns:
        L (array) : eigenvalues of X
        Z (array) : eigenbasis of X


    """
    syevr = la.get_lapack_funcs("syevr", dtype=X.dtype)
    L, Z, _, _, _ = syevr(X)
    return L, Z
