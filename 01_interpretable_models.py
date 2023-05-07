
# %% Imports3rd party libraries first 
# importing all necessary packages here citing XAI christoph Molnar for credits
# The base code is derived from the above and necesssary changes based on my learning has been 
# implemented here. 

from interpret.glassbox import (LogisticRegression,
                                ClassificationTree, 
                                ExplainableBoostingClassifier)
from interpret import show
from sklearn.metrics import f1_score, accuracy_score
from utils import LoadDatafromCSV


# %% Load and preprocess data
load_data = LoadDatafromCSV()
load_data.load_dataset()
load_data.preprocess_data()

# Split the data for evaluation 75 - 25 
X_train, X_test, y_train, y_test = load_data.get_data_split()
print(X_train.shape)
print(X_test.shape)
# Oversample the train data
X_train, y_train = load_data.oversample(X_train, y_train)
print("After oversampling:", X_train.shape)

# %% Fit logistic regression model as this is a binary classfication 
# problem If we use L1 regularization in Logistic Regression all the 
# Less important features will become zero. If we use L2 regularization 
# then the wi values will become small but not necessarily zero 
# liblinear — Library for Large Linear Classification. Uses a coordinate descent algorithm. Coordinate descent is based on minimizing a multivariate function by solving univariate optimization problems in a loop. 
# In other words, it moves toward the minimum in one direction at a time.
# The solvers implemented in the class Logistic 
# Regression are “liblinear”, “newton-cg”, “lbfgs”, “sag” and “saga”.
#X_train.columns = X_train.columns.astype(str)
logisticregress = LogisticRegression(random_state=2021, feature_names=X_train.columns, 
                        penalty='l1', solver='liblinear')
logisticregress.fit(X_train, y_train)
print("Training finished.")

# %% Evaluate logistic regression model
y_pred = logisticregress.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain local prediction
logisticregress_local = logisticregress.explain_local(X_test[:100], y_test[:100], name='Logistic Regression')
show(logisticregress_local)

# %% Explain global logistic regression model
logisticregress_global = logisticregress.explain_global(name='Logistic Regression')
show(logisticregress_global)

# %% Fit decision tree model
classtree = ClassificationTree()
classtree.fit(X_train, y_train)
print("Training finished.")
y_pred = classtree.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain local prediction
classtree_local = classtree.explain_local(X_test[:150], y_test[:150], name='Tree')
show(classtree_local)

# %% Fit Explainable Boosting Machine
explainboostclass = ExplainableBoostingClassifier(random_state=2021)
explainboostclass.fit(X_train, y_train) 
print("Training finished.")
y_pred = explainboostclass.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain locally
explainboostclass_local = explainboostclass.explain_local(X_test[:150], y_test[:150], name='EBM')
show(explainboostclass_local)

# %% Explain globally
explainboostclass_global = explainboostclass.explain_global(name='EBM')
show(explainboostclass_global)
# %%
