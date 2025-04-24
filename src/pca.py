from src.covariance_matrix import covariance_matrix
from src.find_eigenvalues import find_eigenvalues
from src.explained_variance_ratio import explained_variance_ratio
from src.find_eigenvectors import find_eigenvectors
from src.auto_select_k import auto_select_k
from src.Matrix import Matrix


def pca(X: Matrix, k: int = None, threshold: float = 0.95):
    """
    Полный алгоритм PCA с опциональным авто-подбором числа компонент.

    Вход:
      X:       Matrix (n×m) — исходные данные
      k:       int или None — желаемое число компонент
      threshold: float — если k=None, то используется этот порог
                        для auto_select_k(eigenvalues, threshold)
    Выход:
      X_proj: Matrix (n×k) — проекция данных
      gamma:  float        — доля объяснённой дисперсии
      W:      Matrix (m×k) — матрица главных компонент
      means:  list[float]  — вектор средних по колонкам (len=m)
    """
    n, m = X.rows, X.cols
    if n == 0 or m == 0:
        raise ValueError("Пустая матрица X")

    # 1) Вычисляем средние и центрируем
    means = [sum(X.data[i][j] for i in range(n)) / n for j in range(m)]
    X_centered = Matrix([[X.data[i][j] - means[j] for j in range(m)]
                         for i in range(n)])

    # 2) Ковариационная матрица
    C = covariance_matrix(X_centered)

    # 3) Собственные значения и векторы
    eigenvalues = find_eigenvalues(C)
    if not eigenvalues:
        raise ValueError("Не удалось найти собственные значения")
    if k is None:
        k = auto_select_k(eigenvalues, threshold)
    if not (1 <= k <= m):
        raise ValueError(f"k должно быть в диапазоне [1, {m}], получено {k}")

    eigenvectors = find_eigenvectors(C, eigenvalues)
    if len(eigenvectors) < k:
        raise ValueError(f"Найдено лишь {len(eigenvectors)} векторов, запрошено {k}")

    # 4) Сортируем пары (λ, v) и отбираем топ-k
    pairs = list(zip(eigenvalues, eigenvectors))
    pairs.sort(key=lambda x: x[0], reverse=True)
    top_vals, top_vecs = zip(*pairs[:k])

    # 5) Формируем W и проекцию
    W = Matrix((m, k))
    for j, vec in enumerate(top_vecs):
        for i in range(m):
            W.data[i][j] = vec.data[i][0]

    X_proj = X_centered @ W

    # 6) Доля объяснённой дисперсии
    gamma = explained_variance_ratio(eigenvalues, k)

    return X_proj, gamma, W, means

