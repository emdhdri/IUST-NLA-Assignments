import numpy as np


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



def gaussian_elimination(matrix, b):
    n = len(b)
    matrix = np.column_stack((matrix, b))

    for i in range(n):
        max_j = i
        for j in range(i, n):
            if abs(matrix[j][i]) > abs(matrix[max_j][i]):
                max_j = j

        matrix[[i, max_j]] = matrix[[max_j, i]]

        for j in range(i + 1, n):
            m = -matrix[j][i] / matrix[i][i]
            matrix[j] += m * matrix[i]

    return solve_upper_triangular_system(matrix[:, :-1], matrix[:, -1])
