
from helper import Helper

""" Test Code """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
data = pd.read_pickle("./pickle_files/hd_preprossed_data.pkl")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['target']), data.target)

rf = Helper(model, X_train, y_train, X_test, y_test)

accuracy, precision, f1, recall = rf.get_scores()

# rf.plot_roc()
Helper.compareBaselineModels(X_train, y_train, X_test, y_test)
