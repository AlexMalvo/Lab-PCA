from typing import Any, Dict
import random
import math
from src.center_data import center_data
from src.Matrix import Matrix
from src.covariance_matrix import covariance_matrix
from src.find_eigenvalues import find_eigenvalues
from src.auto_select_k import auto_select_k
from src.pca import pca
from src.reconstruction_error import reconstruction_error

def add_noise_and_compare(X: Matrix, noise_level: float = 0.1) -> Dict[str, Any]:
    """
    Добавляет гауссов шум к данным и сравнивает результаты PCA до и после.

    Вход:
      X: матрица данных (n×m)
      noise_level: уровень шума (доля от σ каждого признака)
    Выход:
      {
        "k": int,               # число PCA-компонент
        "orig": {"gamma":…, "mse":…},
        "noisy": {"gamma":…, "mse":…},
        "proj": Matrix,         # проекция исходных (n×k)
        "noisy_proj": Matrix    # проекция зашумлённых (n×k)
      }
    """
    n, m = X.rows, X.cols

    # 1) Авто-подбор k на основе C(X_centered)
    Xc = center_data(X)
    C  = covariance_matrix(Xc)
    ev = find_eigenvalues(C)
    k0 = auto_select_k(ev)
    k  = max(k0, 2)        # минимум 2 компоненты для визуализации

    # 2) PCA на исходных данных
    Xp, gamma0, W, means = pca(X, k)

    # 3) Восстановление и MSE для исходных
    Wt   = W.transpose()
    Xrec = Xp @ Wt
    Xorig_recon = Matrix((n, m))
    for i in range(n):
        for j in range(m):
            Xorig_recon.data[i][j] = Xrec.data[i][j] + means[j]
    mse0 = reconstruction_error(X, Xorig_recon)

    # 4) Стандартные отклонения по столбцам
    stds = []
    for j in range(m):
        col = [X.data[i][j] for i in range(n)]
        mu  = sum(col) / n
        var = sum((v - mu)**2 for v in col) / (n - 1) if n > 1 else 0.0
        stds.append(math.sqrt(var))

    # 5) Формируем зашумлённые данные
    noisy = [
        [X.data[i][j] + random.gauss(0, noise_level * stds[j])
         for j in range(m)]
        for i in range(n)
    ]
    Xn = Matrix(noisy)

    # 6) PCA на зашумлённых (те же k)
    Xpn, gamma1, Wn, means_n = pca(Xn, k)

    # 7) Восстановление и MSE для зашумлённых
    Wnt   = Wn.transpose()
    Xnrec = Xpn @ Wnt
    Xnoisy_recon = Matrix((n, m))
    for i in range(n):
        for j in range(m):
            Xnoisy_recon.data[i][j] = Xnrec.data[i][j] + means_n[j]
    mse1 = reconstruction_error(Xn, Xnoisy_recon)

    return {
        "k": k,
        "orig":  {"gamma": gamma0, "mse": mse0},
        "noisy": {"gamma": gamma1, "mse": mse1},
        "proj": Xp,
        "noisy_proj": Xpn
    }