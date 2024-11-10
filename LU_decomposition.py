import numpy as np



def doolittle_decomposition(matrix):
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input matrix must be square for LU decomposition.")

    dim = shape[0]
    L = np.eye(dim)
    U = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i, dim):
            sum = np.dot(L[i, :i], U[:i, j])
            U[i][j] = matrix[i][j] - sum

        for j in range(i, dim):
            sum = np.dot(L[j, :i], U[:i, i])
            if U[i][i] == 0:
                raise ValueError("Matrix is singular; cannot perform LU decomposition.")
            L[j][i] = (matrix[j][i] - sum) / U[i][i]

    return L, U


def crout_decomposition(matrix):
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input matrix must be square for LU decomposition.")

    dim = shape[0]
    L = np.zeros((dim, dim))
    U = np.eye(dim)

    for i in range(dim):
        for j in range(i, dim):
            sum = np.dot(L[j, :i], U[:i, i])
            L[j][i] = matrix[j][i] - sum

        for j in range(i, dim):
            sum = np.dot(L[i, :i], U[:i, j])
            if L[i][i] == 0:
                raise ValueError("Matrix is singular; cannot perform LU decomposition.")
            U[i][j] = (matrix[i][j] - sum) / L[i][i]

    return L, U


def solve_lower_triangular_system(L, b):
    x = np.zeros(len(b))

    for i in range(len(b)):
        sum = 0
        for j in range(i):
            sum += x[j] * L[i][j]
        x[i] = (b[i] - sum) / L[i][i]

    return x


def solve_upper_triangular_system(U, b):
    n = len(b)
    x = np.zeros(n)

    for i in reversed(range(n)):
        sum = 0
        for j in range(i + 1, n):
            sum += x[j] * U[i][j]
        x[i] = (b[i] - sum) / U[i][i]
        
    return x


def LU_decomposition_system_solver(matrix, b, decomposition_function):
    L, U = decomposition_function(matrix)

    z = solve_lower_triangular_system(L, b)
    x = solve_upper_triangular_system(U, z)
    
    return x

