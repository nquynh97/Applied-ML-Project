from sklearn.neighbors import KNeighborsClassifier

class knn:
    def __init__(self, neighbor):
        self.k_nn = KNeighborsClassifier(n_neighbors=neighbor)

    def fit(self, X_train, y_train):
        self.k_nn.fit(X_train, y_train)

    def predict(self, X_test):
        return self.k_nn.predict(X_test)

    def score(self, X_test, y_test):
        return self.k_nn.score(X_test, y_test)