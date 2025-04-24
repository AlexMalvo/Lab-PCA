from src.Matrix import Matrix
from typing import List
import random
import math

def find_eigenvalues(C: 'Matrix', tol: float = 1e-6) -> List[float]:
    """
    Находит все собственные значения матрицы C методом power iteration с дефляцией.
    Возвращает список собственных значений, упорядоченных по убыванию.
    """
    n = C.rows
    # Копируем данные матрицы
    A = [[C.data[i][j] for j in range(n)] for i in range(n)]
    eigenvalues: List[float] = []
    for _ in range(n):
        # Случайный начальный вектор
        b = [random.random() for _ in range(n)]
        # Нормируем
        norm_b = math.sqrt(sum(x*x for x in b)) or 1.0
        b = [x / norm_b for x in b]
        lambda_old = 0.0
        # Итерации power iteration
        for _ in range(1000):
            # A @ b
            Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
            norm_ab = math.sqrt(sum(x*x for x in Ab)) or 1.0
            b = [x / norm_ab for x in Ab]
            # Rayleigh quotient
            lambda_new = sum(b[i] * sum(A[i][j] * b[j] for j in range(n)) for i in range(n))
            if abs(lambda_new - lambda_old) < tol:
                break
            lambda_old = lambda_new
        eigenvalues.append(lambda_new)
        # Дефляция: A = A - λ * b b^T
        for i in range(n):
            for j in range(n):
                A[i][j] -= lambda_new * b[i] * b[j]
    return eigenvalues
