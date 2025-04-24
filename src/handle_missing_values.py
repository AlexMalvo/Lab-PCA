import math
from src.Matrix import Matrix

def handle_missing_values(X: 'Matrix') -> 'Matrix':
    """
    Вход:
    X: матрица данных (n×m) с возможными значениями NaN
    Выход:
    X_filled: матрица данных (n×m) без NaN (пропуски заменены на среднее по столбцу)
    """
    n, m = X.rows, X.cols
    # Вычисляем среднее по столбцам, игнорируя NaN
    means = [0.0] * m
    counts = [0] * m
    for j in range(m):
        col_sum = 0.0
        count = 0
        for i in range(n):
            val = X.data[i][j]
            if isinstance(val, float) and math.isnan(val):
                continue  # пропуск
            col_sum += val
            count += 1
        means[j] = col_sum / count if count > 0 else 0.0
        counts[j] = count
    # Формируем новую матрицу, заполняя пропуски средними
    X_filled = Matrix((n, m))
    for i in range(n):
        for j in range(m):
            val = X.data[i][j]
            if isinstance(val, float) and math.isnan(val):
                X_filled.data[i][j] = means[j]
            else:
                X_filled.data[i][j] = float(val)
    return X_filled