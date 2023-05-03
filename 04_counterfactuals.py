# %% Imports
from utils import LoadDatafromCSV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import dice_ml

# %% Load and preprocess data
data_loader = LoadDatafromCSV()
data_loader.load_dataset()
data_loader.prep_data()

#numerical = ["age", "hours_per_week"]
#categorical = x_train.columns.difference(numerical)

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

categorical_cols = ["gender",
                            "ever_married",
                            "work_type",
                            "Residence_type",
                            "smoking_status"]
        
numerical = ["age","hypertension",
                     "heart_disease","avg_glucose_level",
                     "bmi"]
        
numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
             ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical),
                ('cat', categorical_transformer, categorical_cols)])

clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"F1 Score ********  {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy ++++++++  {accuracy_score(y_test, y_pred)}")

d = dice_ml.Data(dataframe=data_loader.data, 
                         # For perturbation strategy
                         continuous_features=['age', 
                                              'avg_glucose_level',
                                              'bmi'], 
                         outcome_name='stroke')
backend = 'sklearn'
m = dice_ml.Model(model=model, backend=backend)
exp_random = dice_ml.Dice(d, m, method="random")
query_instances = X_train[4:6]
dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=2, desired_class="opposite", verbose=False)
dice_exp_random.visualize_as_dataframe(show_only_changes=True)

print(f" ++++++++++++++++ {X_test[0:1]} ")

query_instances = X_test[0:1]
dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=2, desired_class="opposite", verbose=False)
dice_exp_random.visualize_as_dataframe(show_only_changes=True)

# %% Create diverse counterfactual explanations


# %% Create feasible (conditional) Counterfactuals


# Selecting the features to vary
# Here, you can ensure that DiCE varies only 
# features that it makes sense to vary. 
# BMI cannot be lower than 15. So a permistted range can be specified

features_to_vary=['avg_glucose_level',
                  'bmi',
                  'smoking_status']
permitted_range={'avg_glucose_level':[50,250],
                'bmi':[18, 35]}

dice_exp_random = exp_random.generate_counterfactuals(
        query_instances, total_CFs=4, desired_class="opposite",
        permitted_range=permitted_range,
        features_to_vary=features_to_vary)

dice_exp_random.visualize_as_dataframe(show_only_changes=True)

"""
# Now generating explanations using the new feature weights
cf = explainer.generate_counterfactuals(input_datapoint, 
                                  total_CFs=3, 
                                  desired_class="opposite",
                                  permitted_range=permitted_range,
                                  features_to_vary=features_to_vary)
# Visualize it
cf.visualize_as_dataframe(show_only_changes=True)

"""
