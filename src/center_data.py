from src.Matrix import Matrix

def center_data(X: 'Matrix') -> 'Matrix':
    if X.rows == 0 or X.cols == 0:
        return Matrix((X.rows, X.cols))
    n, m = X.rows, X.cols
    means = [0.0] * m
    for j in range(m):
        col_sum = sum(X.data[i][j] for i in range(n))
        means[j] = col_sum / n
    X_centered = Matrix((n, m))
    for i in range(n):
        for j in range(m):
            X_centered.data[i][j] = X.data[i][j] - means[j]
    return X_centered