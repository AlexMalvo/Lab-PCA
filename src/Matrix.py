class Matrix:
    def __init__(self, data):
        if isinstance(data, tuple):
            n, m = data
            self.data = [[0.0] * m for _ in range(n)]
        else:
            self.data = [list(map(float, row)) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows > 0 else 0

    def shape(self):
        return (self.rows, self.cols)

    def transpose(self):
        res = Matrix((self.cols, self.rows))
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[j][i] = self.data[i][j]
        return res

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Можно складывать только Matrix с Matrix")
        if self.shape() != other.shape():
            raise ValueError("Размерности матриц не совпадают")
        res = Matrix((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[i][j] = self.data[i][j] + other.data[i][j]
        return res

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Можно вычитать только Matrix из Matrix")
        if self.shape() != other.shape():
            raise ValueError("Размерности матриц не совпадают")
        res = Matrix((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[i][j] = self.data[i][j] - other.data[i][j]
        return res

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Оператор @ доступен только для двух объектов Matrix")
        if self.cols != other.rows:
            raise ValueError("Несогласованные размеры матриц для умножения")
        res = Matrix((self.rows, other.cols))
        for i in range(self.rows):
            for k in range(self.cols):
                a_val = self.data[i][k]
                if a_val != 0.0:
                    for j in range(other.cols):
                        res.data[i][j] += a_val * other.data[k][j]
        return res

    def __repr__(self):
        return "Matrix(" + str(self.data) + ")"

    def __eq__(self, other):
        if not isinstance(other, Matrix) or self.shape() != other.shape():
            return False
        eps = 1e-9
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - other.data[i][j]) > eps:
                    return False
        return True

    def __getitem__(self, idx):
        return self.data[idx]
