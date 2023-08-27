import pandas as pd
import sklearn
from sklearn import preprocessing
import numpy as np

data = pd.read_csv("data/responses.csv")

# data.iloc[0][0:19]
# data.iloc[0][19:31]
# data.iloc[0][31:63]
# data.iloc[0][63:73]
# data.iloc[0][73:76]
# data.iloc[0][76:133]
# data.iloc[0][133:140]
# data.iloc[0][140:151]

# data.loc[1:] = data.loc[:]
# data.loc[0] = new_row

data_numeric = data.select_dtypes(include=['number'])

data_zero_var = data_numeric.copy()

from sklearn.feature_selection import VarianceThreshold

zero_var_filter = VarianceThreshold(threshold=0)

zero_var_filter.fit(data_zero_var)

zero_var_filter.get_support()

filtered_columns = data_zero_var.columns[zero_var_filter.get_support()]

data_continuous = data_numeric[['Age', 'Height', 'Weight']]

normalizer = preprocessing.StandardScaler()
normalizer.fit(data_continuous)

data_normalized = normalizer.transform(data_continuous)
data_normalized = pd.DataFrame(data_normalized, columns = data_continuous.columns)

data_numeric = data.select_dtypes(include=['number'])

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(data_numeric)
data_min_max_scaled = min_max_scaler.transform(data_numeric)

data_min_max_scaled = pd.DataFrame(data_min_max_scaled, columns = data_numeric.columns)


data_min_max_scaled.loc[data_min_max_scaled.isnull().any(axis=1)]

from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer()
data_min_max_scaled_imputed = mean_imputer.fit_transform(data_min_max_scaled)
data_min_max_scaled_imputed = pd.DataFrame(data_min_max_scaled_imputed, columns=data_min_max_scaled.columns)

data_min_max_scaled_imputed

x = []
for index, row in data_min_max_scaled_imputed.iterrows():
  x.append(row)

y = []
for column in data_min_max_scaled_imputed:
  y.append(column)

from scipy.spatial.distance import pdist, squareform

distances = pdist(data_min_max_scaled_imputed.values, metric='euclidean')
dist_matrix = squareform(distances)

# print(dist_matrix)

# distance_list = dist_matrix[0][1:]
# print(distance_list.argmax(), ' : ', distance_list[distance_list.argmax()])
# print(distance_list.argmin(), ' : ', distance_list[distance_list.argmin()])