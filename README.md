# Lab-PCA

## 1. Цели и задачи

В рамках данной лабораторной работы необходимо:

- **Реализовать с нуля** основные шаги алгоритма Principal Component Analysis (PCA):
  1. Центрирование данных  
  2. Вычисление ковариационной матрицы  
  3. Поиск собственных значений и векторов  
  4. Проекция данных на \(k\) главных компонент  
  5. Оценка доли объяснённой дисперсии и ошибки восстановления  
- **Развить и протестировать** сопутствующие модули:
  - Решение СЛАУ методом Гаусса  
  - Автоматический выбор числа компонент по порогу объяснённой дисперсии  
  - Обработка пропущенных значений (mean-imputation)  
- **Провести эксперименты**:
  - Влияние гауссовского шума на качество PCA  
  - Применение PCA к реальным датасетам (`iris`, `wine`, `digits`, `breast_cancer`) и оценка 1-NN accuracy  
- **Доказать математическое обоснование**:
  - Собственные векторы ковариационной матрицы являются оптимальными направлениями PCA  

---

## 2. Структура проекта

```
Lab-PCA/
├── notebooks/
│   ├── easy/
│   │   ├── center_data.ipynb
│   │   ├── covariance_matrix.ipynb
│   │   └── gauss_solver.ipynb
│   ├── normal/
│   │   ├── explained_variance_ratio.ipynb
│   │   ├── find_eigenvalues.ipynb
│   │   └── find_eigenvectors.ipynb
│   ├── hard/
│   │   ├── pca.ipynb
│   │   ├── pca_projection.ipynb
│   │   └── reconstruction_error.ipynb
│   └── expert/
│       ├── add_noise_and_compare.ipynb
│       ├── apply_pca_to_dataset.ipynb
│       ├── auto_select_k.ipynb
│       ├── handle_missing_values.ipynb
│       └── PCA_proof_eigenvectors.ipynb
├── src/
│   ├── Matrix.py
│   ├── center_data.py
│   ├── covariance_matrix.py
│   ├── gauss_solver.py
│   ├── find_eigenvalues.py
│   ├── find_eigenvectors.py
│   ├── explained_variance_ratio.py
│   ├── pca.py
│   ├── plot_pca_projection.py
│   ├── reconstruction_error.py
│   ├── auto_select_k.py
│   ├── handle_missing_values.py
│   ├── add_noise_and_compare.py
│   ├── apply_pca_to_dataset.py
│   └── knn_accuracy.py
└── README.md                  
```

---
