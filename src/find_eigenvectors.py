from typing import List
import random
import math
from src.Matrix import Matrix

def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """
    Вход:
      C: матрица ковариаций (n×n)
      eigenvalues: список собственных значений
    Выход:
      список собственных векторов (каждый — Matrix-столбец)
    """
    n = C.rows
    A = [[C.data[i][j] for j in range(n)] for i in range(n)]
    eigenvectors: List[Matrix] = []
    tol = 1e-6  # допуск для остановки power iteration

    for _ in eigenvalues:
        # случайный стартовый вектор:
        b = [random.random() for _ in range(n)]
        norm_b = math.sqrt(sum(x*x for x in b)) or 1.0
        b = [x / norm_b for x in b]

        lambda_old = 0.0
        # Power iteration
        for _ in range(1000):
            # умножаем A на b:
            Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
            norm_ab = math.sqrt(sum(x*x for x in Ab)) or 1.0
            b = [x / norm_ab for x in Ab]
            lambda_new = sum(b[i] * Ab[i] for i in range(n))
            if abs(lambda_new - lambda_old) < tol:
                break
            lambda_old = lambda_new

        # сохраняем вектор-столбец в формате Matrix:
        vec = Matrix([[b[i]] for i in range(n)])
        eigenvectors.append(vec)

        # дефляция: вычитаем найденное собственное значение из матрицы:
        for i in range(n):
            for j in range(n):
                A[i][j] -= lambda_old * b[i] * b[j]

    return eigenvectors
