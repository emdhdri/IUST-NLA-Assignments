import numpy as np


def jacobi_iterative_solver(matrix, rhs, iterations, initial=None):
    n = len(rhs)
    if initial is None:
        initial = np.zeros(n)
    x = initial

    for i in range(iterations):
        x_tmp = np.empty(n)

        for j in range(n):
            sum = np.dot(matrix[j], x) - x[j] * matrix[j][j]
            x_tmp[j] = (rhs[j] - sum) / matrix[j][j]

        x = x_tmp.copy()

    return x


def gauss_seidel_iterative_solver(matrix, rhs, iterations, initial=None):
    n = len(rhs)
    if initial is None:
        initial = np.zeros(n)
    x = initial

    for i in range(iterations):
        for j in range(n):
            sum = np.dot(matrix[j], x) - x[j] * matrix[j][j]
            x[j] = (rhs[j] - sum) / matrix[j][j]

    return x


def SOR_iterative_solver(matrix, rhs, iterations, relaxation_factor, initial=None):
    n = len(rhs)
    if initial is None:
        initial = np.zeros(n)
    x = initial

    for i in range(iterations):
        for j in range(n):
            sum = np.dot(matrix[j], x) - x[j] * matrix[j][j]
            x[j] = (
                relaxation_factor * (rhs[j] - sum) / matrix[j][j]
                + (1 - relaxation_factor) * x[j]
            )

    return x
