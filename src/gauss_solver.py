from typing import List
from src.Matrix import Matrix


def gauss_solver(A: 'Matrix', b: 'Matrix') -> List['Matrix']:
    """
    Вход:
    A: матрица коэффициентов (n×n). Используется класс Matrix.
    b: вектор правых частей (n×1)
    Выход:
    list[Matrix]: список базисных векторов решения системы
    Raises:
        ValueError: если система несовместна
    """
    n = A.rows
    m = A.cols
    if b.rows != n or b.cols != 1:
        raise ValueError("b должен быть столбцовым вектором длины n")

    # Формируем расширенную матрицу [A|b]
    aug = [A.data[i][:] + [b.data[i][0]] for i in range(n)]
    EPS = 1e-9  # порог для нуля
    pivot_cols = []  # список индексов ведущих столбцов
    row = 0

    # Прямой ход
    for col in range(m):
        # Поиск ненулевого ведущего элемента в текущем столбце
        pivot = row
        while pivot < n and abs(aug[pivot][col]) < EPS:
            pivot += 1
        if pivot == n:
            # В этом столбце ведущего элемента нет
            continue
        # Меняем местами текущую строку и строку с найденным ведущим элементом
        if pivot != row:
            aug[row], aug[pivot] = aug[pivot], aug[row]
        pivot_cols.append(col)
        pivot_val = aug[row][col]
        # Нормируем ведущий элемент к 1 (не обязательно, но удобно)
        if abs(pivot_val - 1.0) > 1e-12:
            for j in range(col, m + 1):
                aug[row][j] /= pivot_val
        # Обнуление элементов ниже ведущего в текущем столбце
        for i in range(row + 1, n):
            factor = aug[i][col]
            if abs(factor) > EPS:
                for j in range(col, m + 1):
                    aug[i][j] -= factor * aug[row][j]
        row += 1
        if row == n:
            break
    # Теперь матрица приведена к ступенчатому виду
    rank = len(pivot_cols)
    # Проверяем на несовместность: если есть строка [0...0 | c], c != 0
    for i in range(rank, n):
        # все коэффициенты 0?
        if all(abs(aug[i][j]) < EPS for j in range(m)) and abs(aug[i][m]) > EPS:
            raise ValueError("Система несовместна")
    solutions: List[Matrix] = []

    # Обратный ход
    x = [0.0] * m
    if rank == m:
        # Единственное решение
        for r in range(m - 1, -1, -1):
            j = pivot_cols[r]  # индекс столбца с ведущим элементом на этой строке
            s = aug[r][m]
            for k in range(j + 1, m):
                s -= aug[r][k] * x[k]
            x[j] = s / (aug[r][j] if abs(aug[r][j]) > EPS else 1.0)
        sol_vec = Matrix([[x[j]] for j in range(m)])  # столбец-матрица решения
        solutions.append(sol_vec)
    else:
        # Бесконечно много решений: параметризуем свободные переменные
        free_cols = [j for j in range(m) if j not in pivot_cols]
        # Находим частное решение (при всех свободных = 0)
        x_part = [0.0] * m
        for r in range(rank - 1, -1, -1):
            j = pivot_cols[r]
            s = aug[r][m]
            for k in range(j + 1, m):
                s -= aug[r][k] * (x_part[k] if k not in free_cols else 0.0)
            x_part[j] = s / (aug[r][j] if abs(aug[r][j]) > EPS else 1.0)
        solutions.append(Matrix([[x_part[j]] for j in range(m)]))

        # Формируем базисные решения, положив по очереди одну свободную переменную = 1, остальные = 0
        for f in free_cols:
            x_h = [0.0] * m
            x_h[f] = 1.0
            for r in range(rank - 1, -1, -1):
                j = pivot_cols[r]
                s = 0.0
                for k in range(j + 1, m):
                    s -= aug[r][k] * (x_h[k])
                x_h[j] = s / (aug[r][j] if abs(aug[r][j]) > EPS else 1.0)
            solutions.append(Matrix([[x_h[j]] for j in range(m)]))
    return solutions
