from typing import List, Any

def knn_accuracy(data: List[List[float]], labels: List[Any]) -> float:
    """
    Простая 1-NN accuracy (leave-one-out).
    """
    n = len(data)
    correct = 0
    for i in range(n):
        best_dist = float('inf')
        best_label = None
        for j in range(n):
            if i == j:
                continue
            # квадрат эвклидова расстояния
            d = sum((data[i][f] - data[j][f])**2 for f in range(len(data[0])))
            if d < best_dist:
                best_dist, best_label = d, labels[j]
        if best_label == labels[i]:
            correct += 1
    return correct / n