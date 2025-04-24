from typing import List

def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
    eigenvalues: список собственных значений
    k: число компонент (первых k собственных значений)
    Выход:
    доля объяснённой дисперсии (от 0 до 1)
    """
    if not eigenvalues:
        return 0.0
    if k < 1 or k > len(eigenvalues):
        raise ValueError("k должен быть от 1 до числа собственных значений")
    vals = sorted(eigenvalues, reverse=True)
    total = sum(vals)
    if total == 0:
        return 0.0
    explained = sum(vals[:k])
    return explained / total