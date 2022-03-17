"""
A helper class that helps reduce all the rewriting of basic ML code
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class Helper:

    _accuracy_scores = {}
    _precision_scores = {}
    _recall_scores = {}
    _f1_scores = {}

    def __init__(self, model, X_train, y_train, X_test, y_test):
        """
        model : strictly a sklearn estimator
        X_train : training data as compaiable with sklearn
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # fit the model
        self.model.fit(X_train, y_train)

        # make the predictions
        self.predictions = self.model.predict(X_test)

        # get the predict proba
        self.predict_proba_values = self.model.predict_proba(X_test)

    def get_scores(self):
        """
        returns the following score metrics : accuracy, precision, recall, f1
        """
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)

        print(f"Fetching results : {self.model}")
        print(
            f"Accuracy :{accuracy}  Precision : {precision}  Recall : {recall}  F1_Score : {f1}\n")

        return accuracy, precision, recall, f1

    def plot_roc(self):
        """
        plots the roc curve based on the y_test and the prediction_proba values
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.predict_proba_values[:, [1]])
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
        return fpr, tpr

    def add_scores(self, scores):
        """
        this function adds the scores to the class variables for finding the best scores
        """
        Helper._accuracy_scores[self.model] = scores[0]
        Helper._precision_scores[self.model] = scores[1]
        Helper._recall_scores[self.model] = scores[2]
        Helper._f1_scores[self.model] = scores[3]

    @classmethod
    def compareBaselineModels(cls, X_train, y_train, X_test, y_test):
        """
        this function returns the scores for few classification models namely : 
        - Decision Tree, RandomForestClassifier, Support Vector Classifier, K Nearest Neighbours, Gradient Boosting
        """
        decisionTree = Helper(DecisionTreeClassifier(
            random_state=100), X_train, y_train, X_test, y_test)
        decisionTreeScores = list(decisionTree.get_scores())
        decisionTree.add_scores(decisionTreeScores)

        randomForest = Helper(RandomForestClassifier(
            random_state=100), X_train, y_train, X_test, y_test)
        randomForestScores = list(randomForest.get_scores())
        randomForest.add_scores(randomForestScores)

        svc = Helper(SVC(probability=True, random_state=100),
                     X_train, y_train, X_test, y_test)
        svcScores = list(svc.get_scores())
        svc.add_scores(svcScores)

        knn = Helper(KNeighborsClassifier(), X_train, y_train, X_test, y_test)
        knnScores = list(knn.get_scores())
        knn.add_scores(knnScores)

        # print the statistics
        print("-"*100)
        highest_accuracy = max(Helper._accuracy_scores,
                               key=Helper._accuracy_scores.get)
        print(
            f"\tHighest Accuracy : {highest_accuracy} : {Helper._accuracy_scores[highest_accuracy]}")

        highest_precision = max(Helper._precision_scores,
                                key=Helper._precision_scores.get)
        print(
            f"\tHighest Precision : {highest_precision} : {Helper._precision_scores[highest_precision]}")

        highest_recall = max(Helper._recall_scores,
                             key=Helper._recall_scores.get)
        print(
            f"\tHighest Recall : {highest_recall} : {Helper._recall_scores[highest_recall]}")

        highest_f1 = max(Helper._f1_scores, key=Helper._f1_scores.get)
        print(f"\tHighest F1 : {highest_f1} : {Helper._f1_scores[highest_f1]}")

        print("-"*100)

        # gradientBoosting = GradientBoosting(random_state=100)
        # knn = KNeighborsClassifier(n_neighbours=3)

    def __str__(self):
        return f"Model : {self.model}\n"
