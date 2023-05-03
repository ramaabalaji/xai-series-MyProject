# %% Imports All necessary imports are added here 
import matplotlib.pyplot as plt
from utils import LoadDatafromCSV

# %% Load data
data_loader = LoadDatafromCSV()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print(data.shape)
data.head()

# %% Show general statistics
data.info()

# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()

# %% Show preprocessed dataframe
data_loader.preprocess_data()
data_loader.data.head()


