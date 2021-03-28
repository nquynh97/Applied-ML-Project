from sklearn.tree import DecisionTreeClassifier


class decisiontree:
    def __init__(self, criterion, max_depth):
        self.decision_tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)

    def fit(self, X_train, y_train):
        self.decision_tree.fit(X_train, y_train)

    def predict(self, X_test):
        return self.decision_tree.predict(X_test)

    def score(self, X_test, y_test):
        return self.decision_tree.score(X_test, y_test)
