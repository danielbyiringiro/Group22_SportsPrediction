#!/usr/bin/env python
# coding: utf-8

# Importing all the external modules that will be used in training, testing, evaluating, visualising, and saving the model.

# In[1]:


import pandas as pd
import pickle
import zipfile
import os
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


# Reading the training dataset from the csv file and storing it in a dataframe.

# In[2]:


dataset = pd.read_csv('players_21.csv')
dataset.head()


# Spliting the dataset into numerical features and categorical features based on the data type of the features.

# In[3]:


dataset.info()

numerical_features = dataset.select_dtypes(include=['int64', 'float64'])
categorical_features = dataset.select_dtypes(include=['object'])

numerical_features.head()


# Dropping columns with more than 30% missing values for the numerical features.

# In[4]:


numerical_features = numerical_features.dropna(thresh=0.7*len(numerical_features), axis=1)
numerical_features.head()


# In[5]:


categorical_features.head()


# Dropping columns with more than 30% missing values for the categorical features.

# In[6]:


categorical_features.info()


categorical_features = categorical_features.dropna(thresh=0.7*len(categorical_features), axis=1)
categorical_features.head()


# Converting the column `dob` to `cat_age` by subtracting the year of birth from the current year. The feature is called `cat_age` because the is an already existing feature called `age` in the numerical features. This is because age has more relational value than the year of birth.
# 
# Converting the column `club_joined` to `cat_years_in_club` by subtracting the year of joining the club from the current year.
# This is because the number of years in the club has more relational value than the year of joining the club.

# In[7]:


categorical_features['dob'] = pd.to_datetime(categorical_features['dob'])
categorical_features['club_joined'] = pd.to_datetime(categorical_features['club_joined'])

categorical_features['cat_age'] = categorical_features['dob'].apply(lambda x: 2023 - x.year)
categorical_features['cat_years_in_club'] = categorical_features['club_joined'].apply(lambda x: 2023 - x.year)

categorical_features = categorical_features.drop(['dob', 'club_joined'], axis=1)
categorical_features.head()


# Dropping columns with more 90% unique values since unique categorical values do not provide any pattern or trends the model can learn from.

# In[8]:


mostly_unique = [col for col in categorical_features.columns if categorical_features[col].nunique() >= 0.9 * len(categorical_features)]
categorical_features = categorical_features.drop(mostly_unique, axis=1)
categorical_features.head()


# Converting columns with categorical values like `89 + 3` to a numerical value like `92` this is done to increase the number of numerical features in the dataset.

# In[9]:


# these columns start from column 9 to column 36

for col in categorical_features.columns[9:36]:
    categorical_features[col] = categorical_features[col].apply(lambda x: eval(x) if '+' in x or '-' in x else int(x))

categorical_features.head()


# Since we have the club_name and nationality of a player we can remove the the club_logo_url, club_flag_url and national_flag_url columns.

# In[10]:


categorical_features = categorical_features.drop(['club_logo_url', 'club_flag_url', 'nation_flag_url'], axis=1)
categorical_features.head()


# Convert the remaining categorical features to numerical by factorizing.

# In[11]:


encodings_map  = {}

for col in categorical_features.select_dtypes(include=['object']).columns:
    encoded_values, unique_categories = pd.factorize(categorical_features[col])
    encodings_map[col] = dict(zip(unique_categories, encoded_values))
    categorical_features[col] = encoded_values

categorical_features.head()


# Combine the numerical and categorical features to form one dataframe that will be used for training the model.

# In[12]:


dataset = pd.concat([numerical_features, categorical_features], axis=1)
dataset.head()


# Drop the `age` feature in the dataset since we have the `cat_age` which is their accurate age as of now. (2023 - dob)

# In[13]:


dataset = dataset.drop('age', axis=1)


# Measure feature importance using the Random Forest Regressor.

# In[14]:


X = dataset.drop(['overall'], axis=1)
y = dataset['overall']


# Imputing the missing values in the dataset by filling them with the mean of the column.

# In[15]:


X_imputed = X.fillna(X.mean())
y_imputed = y

X_imputed.shape, y_imputed.shape


# Training the random forest regressor with a 100 estimators.

# In[16]:


imputedRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
imputedRegressor.fit(X_imputed, y_imputed)


# Showing the feature importance of the model in a dataframe format.

# In[17]:


imputed_feature_importances = pd.DataFrame(imputedRegressor.feature_importances_, index=X_imputed.columns, columns=['importance']).sort_values('importance', ascending=False)
imputed_feature_importances *= 100
imputed_feature_importances.head()


# Keeping only the important features for our X (Independent Variables).

# In[18]:


y = y_imputed

X = imputed_feature_importances[imputed_feature_importances['importance'] > 1].index
X = X_imputed[X]
X.head()


# Checking to see how the data is distributed using X.describe().

# In[19]:


X.describe()


# Scaling the data using the StandardScaler, this is done to make sure that all the features are on the same scale and to avoid some features dominating the others. The StandardScaler object is saved for later use.

# In[20]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler_model.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
X.describe()


# Splitting the data for training and testing using the train_test_split function from sklearn. We want to choose the model to use, so we will use the train data to train the model and the test data to test the model on different models and choose the one that performs best.

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Training the models with 16 different regression models and comparing their performance

# In[22]:


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


# Converting perfomance data into a dataframe and sorting the data using the R2 score metric.

# In[23]:


accuracy = pd.DataFrame([mse, mae, r2], index=['MSE', 'MAE', 'R2']).T
accuracy = accuracy.sort_values('R2', ascending=False)
accuracy


# Plot the accuracy of the models

# In[24]:


plt.figure(figsize=(20, 10))
plt.plot(accuracy['MSE'], label='MSE')
plt.plot(accuracy['MAE'], label='MAE')
plt.plot(accuracy['R2'], label='R2')
plt.xticks(rotation=90)
plt.legend()
plt.show()


# Random forest has the lowest mean absolute error and the highest r2 score so we wil go with the Random Forest Regressor algotithm.
# 
# We now have to tune the hyperparameters of the model to get the best performance, and train it on the entire training dataset.

# In[25]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)

# Combine x_train and x_test into one data frame since its all part of the training set, applying the same to y

X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

grid_search.fit(X, y)

grid_search.best_params_


# Training the model using the best parameters and the entire training dataset.

# In[ ]:


model_with_best_params = RandomForestRegressor(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=500, n_jobs=-1, random_state=42)
model_with_best_params.fit(X, y)


# Reading the test dataset from a csv file and storing it in a dataframe.

# In[ ]:


test_data = pd.read_csv('players_22.csv')
test_data.head()


# Calculating the current age of the players in the test dataset. This corresponds to the `cat_age` feature in the training dataset.

# In[ ]:


test_data['cat_age'] = 2023 - pd.to_datetime(test_data.dob).dt.year
test_data.cat_age.head()


# Creating a subset of the data that contains only the columns that were used to train the model.

# In[ ]:


needed_columns = ['value_eur','release_clause_eur','cat_age','potential','movement_reactions']
test_features = test_data[needed_columns]
test_features.head()


# In[ ]:


test_features.info()


# In[ ]:


test_overall = test_data.overall
test_overall.head()


# Imputing the missing values in the dataset by filling them with the mean of the column.

# In[ ]:


test_features = test_features.fillna(test_features.mean())


# In[ ]:


test_features.head()


# Scaling the data using the StandardScaler object that was saved earlier.

# In[ ]:


with open('scaler_model.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)  # Load the saved scaler
test_features = loaded_scaler.transform(test_features)


# Predicting the overall rating of the players in the test dataset.

# In[ ]:


y_pred = model_with_best_params.predict(test_features)


# Printing the metrics of the model.

# In[ ]:


mae = mean_absolute_error(test_overall, y_pred)
r2 = r2_score(test_overall, y_pred)
mse = mean_squared_error(test_overall, y_pred)

print(f'MAE: {mae}')
print(f'R2: {r2}')
print(f'MSE: {mse}')


# Plotting the actual and predicted values to see the correlation

# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(test_overall, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Player')
plt.ylabel('Overall')
plt.legend()
plt.show()


# In[ ]:


comparison = pd.DataFrame({'Actual': test_overall, 'Predicted': y_pred})
comparison.tail(10)


# The model has an r2 score of 0.97 which is very good, and the mean absolute error is 0.5 which is also very good.

# Saving the model for deployment in the web app using pickle.

# In[ ]:


with open('model.pkl', 'wb') as f:
    pickle.dump(model_with_best_params, f)


# The raw model is very large (more than 300mb in size) so we will compress it using gzip to make it smaller (less than 100mb) in size. This will allow us to upload it to github.

# In[ ]:


zip_filename = "model.zip"
model_filename = "model.pkl"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(model_filename)

print(f"{model_filename} has been zipped to {zip_filename}")


# Deleting the raw model to save space.

# In[ ]:


file_path = 'model.pkl'

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been removed")

