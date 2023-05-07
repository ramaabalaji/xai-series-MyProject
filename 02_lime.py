# %% Imports
# %% Imports3rd party libraries first 
# importing all necessary packages here citing XAI christoph Molnar for credits
# The base code is derived from the above and necesssary changes based on my learning has been 
# implemented here. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import lime
import lime.lime_tabular
from lime import submodular_pick

import warnings
warnings.filterwarnings('ignore')
from interpret.blackbox import LimeTabular
from interpret import show

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from utils import LoadDatafromCSV


# %% Load and preprocess data
data_loader = LoadDatafromCSV()
data_loader.load_dataset()
data_loader.preprocess_data_dummy()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Oversample the minority data is an option - train data

# Just uncomment if you wish to only use 
# oversampling of the minority data 


#X_train, y_train = data_loader.oversample(X_train, y_train)
#print(X_train.shape)
#print(X_test.shape)
#print(f'Printed X_train and X_test')

# The best way is to use SMOTE for sampling the data. 
# Pipeline has been used to combine, SMOTE - oversampling for minority data and 
# undersampling for the majority class. 

X_train, y_train = data_loader.smotesample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)
print("After smotesampling: ---------------------")


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
exp = explainer.explain_instance(X_test.values[12], rf.predict, num_features=10)
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





model = XGBClassifier(n_estimators = 300, random_state = 123)
model.fit(X_train, y_train)
model.score(X_test, y_test)
predict_fn = lambda x: model.predict_proba(x)
np.random.seed(123)
# Defining the LIME explainer object

feature_names=X_train.columns.astype(str).values.tolist(),
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   mode='classification',
                                                   class_names=['No Stroke', 'Stroke'],
                                                   training_labels='Stroke',
                                                   feature_names=feature_names)
# Local Interpretability on particular instance
# using LIME to get the explanations
i = 12
exp = explainer.explain_instance(X_train.loc[i,X_train.columns].astype(str).values, predict_fn, num_features=21)
exp.show_in_notebook(show_table=True)# %%


"""
