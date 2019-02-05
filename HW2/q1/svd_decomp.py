from scipy import linalg
import numpy as np

def compute_svd():
    M = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    U, S, Vt = linalg.svd(M, full_matrices=False)
    Evals, Evecs = linalg.eigh(M.T.dot(M))
    Evals = sorted(Evals, key=int, reverse=True)
    Evecs[:, [1, 0]] = Evecs[:, [0, 1]]

    print("U = ", U)
    print("S = ", S)
    print("Vt = ", Vt)
    print("Evals = ", Evals)
    print("Evecs = ", Evecs)

compute_svd()