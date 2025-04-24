from typing import List

def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """
    Вход:
    eigenvalues: список собственных значений (не обязательно отсортирован)
    threshold: порог объяснённой дисперсии (доля, 0 < threshold <= 1)
    Выход:
    оптимальное число главных компонент k, при котором суммарная объясненная дисперсия >= threshold
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold должен быть в диапазоне (0, 1]")
    if not eigenvalues:
        return 0
    vals = sorted(eigenvalues, reverse=True)
    total = sum(vals)
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += v
        if cum / total >= threshold:
            return i
    return len(vals)
