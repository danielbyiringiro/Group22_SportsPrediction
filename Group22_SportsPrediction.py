#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from matplotlib import pyplot as plt


# In[4]:


dataset = pd.read_csv('players_21.csv')
dataset.head()


# In[5]:


dataset.info()

# split the dataset into numerical and categorical features

numerical_features = dataset.select_dtypes(include=['int64', 'float64'])
categorical_features = dataset.select_dtypes(include=['object'])

numerical_features.head()


# In[6]:


# drop columns with more than 30% missing values

numerical_features = numerical_features.dropna(thresh=0.7*len(numerical_features), axis=1)
numerical_features.head()


# In[7]:


# dealing with categorical features

categorical_features.head()


# In[8]:


categorical_features.info()

# drop columns with more than 30% missing values

categorical_features = categorical_features.dropna(thresh=0.7*len(categorical_features), axis=1)
categorical_features.head()


# In[9]:


# convert dob and club_joined to age and years in club respectively

categorical_features['dob'] = pd.to_datetime(categorical_features['dob'])
categorical_features['club_joined'] = pd.to_datetime(categorical_features['club_joined'])

categorical_features['cat_age'] = categorical_features['dob'].apply(lambda x: 2023 - x.year)
categorical_features['cat_years_in_club'] = categorical_features['club_joined'].apply(lambda x: 2023 - x.year)

categorical_features = categorical_features.drop(['dob', 'club_joined'], axis=1)
categorical_features.head()


# In[10]:


#dropping columns with > 90% unique values since unique categorical values do not provide any pattern or trends the model can learn from

mostly_unique = [col for col in categorical_features.columns if categorical_features[col].nunique() >= 0.9 * len(categorical_features)]
categorical_features = categorical_features.drop(mostly_unique, axis=1)
categorical_features.head()


# In[11]:


# converting columns with categorical values like 89 + 3 to numerical values like 92
# this is done to increase the number of numerical features in the dataset

# these columns start from column 9 to column 36

for col in categorical_features.columns[9:36]:
    categorical_features[col] = categorical_features[col].apply(lambda x: eval(x) if '+' in x or '-' in x else int(x))

categorical_features.head()


# In[12]:


# since we have the club_name and nationality of a player we can remove the the club_logo_url, club_flag_url and national_flag_url columns

categorical_features = categorical_features.drop(['club_logo_url', 'club_flag_url', 'nation_flag_url'], axis=1)
categorical_features.head()


# In[13]:


# convert the remaining categorical features to numerical by factorizing

encodings_map  = {}

for col in categorical_features.select_dtypes(include=['object']).columns:
    encoded_values, unique_categories = pd.factorize(categorical_features[col])
    encodings_map[col] = dict(zip(unique_categories, encoded_values))
    categorical_features[col] = encoded_values

categorical_features.head()


# In[14]:


# combine the numerical and categorical features

dataset = pd.concat([numerical_features, categorical_features], axis=1)
dataset.head()


# In[15]:


# drop age in the dataset since we have cat_age (2023 - dob)

dataset = dataset.drop('age', axis=1)


# In[16]:


# measure feature importance

X = dataset.drop(['overall'], axis=1)
y = dataset['overall']


# In[17]:


# check whether a dataset with dropped values is better than one with imputed values 
# in terms of a distributed feature importance set

X_dropped = X.dropna(how='any', axis=0)
y_dropped = y[X_dropped.index]

X_dropped.shape, y_dropped.shape


# In[18]:


X_imputed = X.fillna(X.mean())
y_imputed = y

X_imputed.shape, y_imputed.shape


# In[19]:


droppedRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
droppedRegressor.fit(X_dropped, y_dropped)

imputedRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
imputedRegressor.fit(X_imputed, y_imputed)


# In[20]:


dropped_feature_importances = pd.DataFrame(droppedRegressor.feature_importances_, index=X_dropped.columns, columns=['importance']).sort_values('importance', ascending=False)
dropped_feature_importances *= 100
dropped_feature_importances.head()


# In[21]:


imputed_feature_importances = pd.DataFrame(imputedRegressor.feature_importances_, index=X_imputed.columns, columns=['importance']).sort_values('importance', ascending=False)
imputed_feature_importances *= 100
imputed_feature_importances.head()


# In[22]:


# move forward with the imputed dataset since it is more distributed in terms of feature importance

# scale the dataset

y = y_imputed

# keeping only the important features

X = imputed_feature_importances[imputed_feature_importances['importance'] > 1].index
X = X_imputed[X]
X.head()


# In[23]:


X.describe()


# In[24]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler_model.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
X.describe()


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[26]:


# training the models with 16 different regression models and comparing their performance

regression_models = {
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SGDRegressor': SGDRegressor(),
    'BayesianRidge': BayesianRidge(),
    'HuberRegressor': HuberRegressor(),
    'SVR': SVR(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'MLPRegressor': MLPRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'ExtraTreesRegressor': ExtraTreesRegressor(),
    'XGBRegressor': XGBRegressor()
}

mse = {}
mae = {}
r2 = {}

for name, model in regression_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse[name] = mean_squared_error(y_test, y_pred)
    mae[name] = mean_absolute_error(y_test, y_pred)
    r2[name] = r2_score(y_test, y_pred)


# In[27]:


accuracy = pd.DataFrame([mse, mae, r2], index=['MSE', 'MAE', 'R2']).T
accuracy = accuracy.sort_values('R2', ascending=False)
accuracy


# In[28]:


# plot the accuracy of the models

plt.figure(figsize=(20, 10))
plt.plot(accuracy['MSE'], label='MSE')
plt.plot(accuracy['MAE'], label='MAE')
plt.plot(accuracy['R2'], label='R2')
plt.xticks(rotation=90)
plt.legend()
plt.show()


# In[29]:


# Random forest has the lowest mean absolute error and the highest r2 score
# so we wil go with Random Forest


# In[ ]:


# hyperparameter tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)

# combine x_train and x_test into one data frame since its all part of the training set, applying the same to y

X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

grid_search.fit(X, y)

grid_search.best_params_


# In[31]:


model_with_best_params = RandomForestRegressor(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=500, n_jobs=-1, random_state=42)
model_with_best_params.fit(X, y)


# In[32]:


test_data = pd.read_csv('players_22.csv')
test_data.head()


# In[33]:


test_data['cat_age'] = 2023 - pd.to_datetime(test_data.dob).dt.year
test_data.cat_age.head()


# In[34]:


needed_columns = ['value_eur','release_clause_eur','cat_age','potential','movement_reactions']
test_features = test_data[needed_columns]
test_features.head()


# In[35]:


test_features.info()


# In[36]:


test_overall = test_data.overall
test_overall.head()


# In[37]:


# drop missing features in test features

test_features = test_features.fillna(test_features.mean())


# In[38]:


test_features.head()


# In[39]:


with open('scaler_model.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)  # Load the saved scaler
test_features = loaded_scaler.transform(test_features)


# In[40]:


y_pred = model_with_best_params.predict(test_features)


# In[41]:


mae = mean_absolute_error(test_overall, y_pred)
r2 = r2_score(test_overall, y_pred)
mse = mean_squared_error(test_overall, y_pred)

print(f'MAE: {mae}')
print(f'R2: {r2}')
print(f'MSE: {mse}')


# In[42]:


# plot the actual and predicted values

plt.figure(figsize=(20, 10))
plt.plot(test_overall, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Player')
plt.ylabel('Overall')
plt.legend()
plt.show()


# In[43]:


comparison = pd.DataFrame({'Actual': test_overall, 'Predicted': y_pred})
comparison.tail(10)


# In[44]:


# its better at predicting higher overall players than lower overall players

# save the model

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model_with_best_params, f)


# In[47]:


import zipfile

zip_filename = "model.zip"
model_filename = "model.pkl"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(model_filename)

print(f"{model_filename} has been zipped to {zip_filename}")


# In[ ]:


import os
file_path = 'model.pkl'

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been removed")

