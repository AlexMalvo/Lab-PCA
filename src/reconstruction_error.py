def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    n, m = X_orig.rows, X_orig.cols
    if X_recon.rows != n or X_recon.cols != m:
        raise ValueError("Размерности X_orig и X_recon должны совпадать")
    if n == 0 or m == 0:
        return 0.0
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            diff = X_orig.data[i][j] - X_recon.data[i][j]
            total_error += diff * diff
    return total_error / (n * m)
