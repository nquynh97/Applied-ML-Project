from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class randomforest:
    def __init__(self, n_estimators, criterion, max_depth):
        self.random_forest = RandomForestClassifier(n_estimators=n_estimators,
                                                    criterion=criterion,
                                                    max_depth=max_depth,
                                                    random_state=None)
        self.tree = DecisionTreeClassifier(criterion, max_depth, random_state=0)

    def fit(self, X_train, y_train):
        self.random_forest.fit(X_train, y_train)
        
    def tree_fit(self, X_train, y_train):
        self.tree.fit(X_train, y_train)

    def predict(self, X_test):
        return self.random_forest.predict(X_test)

    def score(self, X_test, y_test):
        return self.random_forest.score(X_test, y_test)
    
    def tree_score(self, X_test, y_test):
        return self.tree.score(X_test, y_test)
