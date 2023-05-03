# %% Imports
# importing all necessary packages here citing XAI christoph Molnar for credits

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

import lime
import lime.lime_tabular

from interpret.blackbox import LimeTabular
from interpret import show
from utils import LoadDatafromCSV

# %% Load and preprocess data
data_loader = LoadDatafromCSV()
data_loader.load_dataset()
data_loader.preprocess_data_dummy()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)
print('Printed X_train and X_test')

# %% Fit blackbox model
rf = RandomForestClassifier()
X_train.columns = X_train.columns.astype(str)
rf.fit(X_train.values, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Apply lime
# Initilize Lime for Tabular data
# predict_fn=rf.predict_proba, 

# LIME has one explainer for all the models
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, 
    feature_names=X_train.columns.astype(str).values.tolist(),
    class_names=['MEDV'], verbose=True, mode='regression')
exp = explainer.explain_instance(X_test.values[10], rf.predict, num_features=10)
exp.show_in_notebook(show_table=True)
exp.as_list()

"""
lime = LimeTabular(model=rf.predict_proba,
                   data=X_train, 
                   random_state=1)
# Get local explanations
lime_local = lime.explain_local(X_test[-20:], 
                                y_test[-20:], 
                                name='LIME')

show(lime_local)



"""


# %%
