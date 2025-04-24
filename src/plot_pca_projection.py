from src.Matrix import Matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_pca_projection(X_proj: 'Matrix') -> Figure:
    """
    Вход: проекция данных X_proj (n×2) – двумерное представление данных
    Выход: объект Figure с графиком рассеяния точек в новом пространстве
    """
    if not isinstance(X_proj, Matrix) or X_proj.cols != 2:
        raise ValueError("Для визуализации размерность проекции должна быть 2 и тип Matrix")
    x_coords = [X_proj.data[i][0] for i in range(X_proj.rows)]
    y_coords = [X_proj.data[i][1] for i in range(X_proj.rows)]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_coords, y_coords, s=10, alpha=0.7)
    ax.set_title("Проекция данных на 2 главные компоненты")
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    return fig
