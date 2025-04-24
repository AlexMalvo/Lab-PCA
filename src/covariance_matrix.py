from src.Matrix import Matrix

def covariance_matrix(X_centered: 'Matrix') -> 'Matrix':
    if X_centered.rows == 0 or X_centered.cols == 0:
        return Matrix((X_centered.cols, X_centered.cols))
    n, m = X_centered.rows, X_centered.cols
    denom = n - 1 if n > 1 else 1
    X_T = X_centered.transpose()
    C = X_T @ X_centered
    for i in range(C.rows):
        for j in range(C.cols):
            C.data[i][j] /= denom
    return C