
from sklearn.model_selection import train_test_split
from helper import Helper

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

"""
Scenario 1
- Create a sklearn model object as usual for a particular model
- Crete a Helper class object
- Call the get_scores() method on the Helper class object
- Call the plot_roc() method to plot
"""

# creating a sklearn model object from RandomFroestClassifier()
model = RandomForestClassifier()
data = pd.read_pickle("./pickle_files/hd_preprossed_data.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['target']), data.target)

# creating a Helper class object
rf = Helper(model, X_train, y_train, X_test, y_test)

# calling the get_scores() method
accuracy, precision, f1, recall = rf.get_scores()
print(accuracy, precision, f1, recall)

# plot the roc curve with plot_roc()
rf.plot_roc()

""" 
Scenario 2
- Simply call the compareBaslineModels() class method with the train and test data
- Good to see the results from all different models at once, and get the score from the best model
"""
Helper.compareBaselineModels(X_train, y_train, X_test, y_test)
