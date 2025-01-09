from collections import Counter
import numpy as np
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, x):
        # Hitung jarak ke semua sampel pelatihan
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ambil k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = [self.y_train[i] for i in k_indices]
        # Tentukan kelas berdasarkan mayoritas
        most_common = Counter(k_neighbors).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return [self._predict(x) for x in np.array(X)]