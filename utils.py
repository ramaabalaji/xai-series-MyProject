# importing all necessary packages here citing XAI christoph Molnar for credits
# The base code is derived from the above and necesssary changes based on my learning has been 
# implemented here. 

import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


class LoadDatafromCSV():

    #dander method used as a constructor
    def __init__(self):
        self.data = None

# Loading Data from the csv file from kaggle

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)

    def prep_data(self):

        # Begin the EDa process by learning about your data 
        # The data contains categorical values which should be converted 
        # before we perform any operations on it. We use One hot encoding 
        # to these variables for simplicity  
        # One-hot encode all categorical columns
        
        # Impute missing values of BMI with 0 if it is not provided already.
        self.data.bmi = self.data.bmi.fillna(0)
        
    def preprocess_data(self):

        # Begin the EDa process by learning about your data 
        # The data contains categorical values which should be converted 
        # before we perform any operations on it. We use One hot encoding 
        # to these variables for simplicity  
        # One-hot encode all categorical columns
        categorical_cols = ["gender",
                            "ever_married",
                            "work_type",
                            "Residence_type",
                            "smoking_status"]
        
        ohe = OneHotEncoder(categories='auto', 
                    drop='first',sparse_output=False)
        
        encoded = pd.DataFrame(ohe.fit_transform(self.data[categorical_cols]))        

        #encoded = pd.get_dummies(self.data[categorical_cols], 
        #                    prefix=categorical_cols)

        # Add the encoded values back into the dataframe and 
        # drop the first value out of the dataframe in order to reduce the correlations 
        # Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)

        # Impute missing values of BMI with 0 if it is not provided already.
        self.data.bmi = self.data.bmi.fillna(0)
        
        # Drop id as it is not relevant
        self.data.drop(["id"], axis=1, inplace=True)


        # Standardization 
        # Usually we would standardize here and convert it back later 
        # This is done in order to keep all parameters to same scale so model does not get baised  
        # But for simplification we will not standardize / normalize the features

    def preprocess_data_dummy(self):

        # Begin the EDa process by learning about your data 
        # The data contains categorical values which should be converted 
        # before we perform any operations on it. We use One hot encoding 
        # to these variables for simplicity  
        # One-hot encode all categorical columns
        categorical_cols = ["gender",
                            "ever_married",
                            "work_type",
                            "Residence_type",
                            "smoking_status"]
        
        encoded = pd.get_dummies(self.data[categorical_cols], 
                            prefix=categorical_cols)

        # Add the encoded values back into the dataframe and 
        # drop the first value out of the dataframe in order to reduce the correlations 
        # Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)

        # Impute missing values of BMI with 0 if it is not provided already.
        self.data.bmi = self.data.bmi.fillna(0)
        
        # Drop id as it is not relevant
        self.data.drop(["id"], axis=1, inplace=True)

        # Standardization 
        # Usually we would standardize here and convert it back later 
        # This is done in order to keep all parameters to same scale so model does not get baised  
        # But for simplification we will not standardize / normalize the features


    def get_data_split(self):
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]

        # 75 25 split for training and testing 
        return train_test_split(X, y, test_size=0.25, random_state=2021)
    
    # Since the stoke data is an imbalanced data set
    # we are performing oversmapling here for the minortity class. 
    # We could also do SMOTE instead of oversampling
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over
    

# The best way is to use SMOTE for sampling the data. 
# Pipeline has been used to combine, SMOTE - oversampling for minority data and 
# undersampling for the majority class. 


    def smotesample(self, X_train, y_train):
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        # transform the dataset
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = pipeline.fit_resample(x_np, y_np)
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over


