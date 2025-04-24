from typing import Tuple
from src.knn_accuracy import knn_accuracy
from src.Matrix import Matrix
from src.pca import pca
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer


def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple[Matrix, float]:
    """
    Загружает встроенный датасет sklearn по имени, применяет вашу PCA(k),
    вычисляет 1-NN accuracy до и после снижения размерности и возвращает
    матрицу проекции и accuracy после.

    Вход:
      dataset_name: 'iris', 'wine', 'digits' или 'breast_cancer'
      k: число главных компонент для PCA
    Выход:
      (X_proj, acc_after) — Matrix n×k и float accuracy после PCA
    """
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'digits':
        data = load_digits()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Неизвестный датасет '{dataset_name}'")

    X_raw = [list(map(float, row)) for row in data.data]
    y     = list(data.target)

    # Accuracy до PCA:
    acc_before = knn_accuracy(X_raw, y)
    print(f"1-NN accuracy до PCA: {acc_before:.4f}")

    X_mat = Matrix(X_raw)
    X_proj, gamma, W, means = pca(X_mat, k)
    print(f"Объяснённая дисперсия (γ) = {gamma:.4f}")

    proj_list = [[X_proj.data[i][j] for j in range(k)] for i in range(X_proj.rows)]
    acc_after = knn_accuracy(proj_list, y)
    print(f"1-NN accuracy после PCA: {acc_after:.4f}")

    return X_proj, acc_after