import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.neighbors import KNeighborsRegressor

#Loading the dataset
df=pd.read_csv('Housing.csv')
print(df.head())

#Inspecting the dataset
print(df.info())


#Handling missing values
print(df.isnull().sum())

#Visualizing the data
plt.hist(df['price'],bins=20)
plt.xlabel('Price')
plt.ylabel('Frequency') 
plt.title('Distribution of House Prices')
plt.show()

x=df.drop('price',axis=1)
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#Encoding categorical variables
for col in x_train.select_dtypes(include=['object']):
    le=LabelEncoder()
    x_train[col]=le.fit_transform(x_train[col])
    x_test[col]=le.transform(x_test[col])


                    #Random Forest Regressor

#Training the model
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

prediction=model.predict(x_test)


#Evaluating the model
mae=mean_absolute_error(y_test,prediction)
print(f"Random Forest MAE: {mae}")

#feature scaling
scaler=StandardScaler()
x_train_scaled,x_test_scaled=scaler.fit_transform(x_train),scaler.transform(x_test)


#feature importance
importances=model.feature_importances_
features=x_train.columns

indices=np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(x_train.shape[1]),importances[indices],align='center')
plt.xticks(range(x_train.shape[1]),features[indices],rotation=90)
plt.tight_layout()
plt.show()

param_grid={
    "n_estimators":[100,200],
    "max_depth":[5,10,20],
    "min_samples_split":[2,5,10]}

grid=GridSearchCV(RandomForestRegressor(random_state=42),
                  param_grid,
                  cv=3,
                  scoring="neg_mean_absolute_error")

grid.fit(x_train_scaled,y_train)
print(f"Best Parameters: {grid.best_params_}")
print(f"Best Score: {-grid.best_score_}")


          #Gradient Boosting Regressor


#Training the model
gb=GradientBoostingRegressor()
gb.fit(x_train_scaled,y_train)
gb_pred=gb.predict(x_test_scaled)

#Evaluating the model
gb_mae=mean_absolute_error(y_test,gb_pred)
print(f"Gradient Boosting MAE: {gb_mae}")

#Hyperparameter tuning for Gradient Boosting
param_grid={
    "n_estimators":[100,200,300],
    "learning_rate":[0.01,0.05,0.1],
    "max_depth":[3,5,10]
}

grid=GridSearchCV(GradientBoostingRegressor(),
                  param_grid,
                  cv=3,
                  scoring="neg_mean_absolute_error")

grid.fit(x_train_scaled,y_train)
print(f"Best Parameters: {grid.best_params_}")
print(f"Best Score: {-grid.best_score_}")

         #K-Nearest Neighbors Regressor

knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train_scaled,y_train)
knn_pred=knn.predict(x_test_scaled)

#Evaluating the model
knn_mae=mean_absolute_error(y_test,knn_pred)
print(f"KNN MAE: {knn_mae}")



best_model=grid.best_estimator_
pred=best_model.predict(x_test_scaled)
final_mae=mean_absolute_error(y_test,pred)
print(f"Best Model MAE: {final_mae}")


# Best hyperparameters from GridSearchCV
best_params = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}

# Build the model
best_gb_model = GradientBoostingRegressor(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    random_state=42
)

# Fit on training data
best_gb_model.fit(x_train_scaled, y_train)

# Predict on test data
pred = best_gb_model.predict(x_test_scaled)

# Evaluate the model
final_mae = mean_absolute_error(y_test, pred)
final_r2 = r2_score(y_test, pred)
print(f"Best Gradient Boosting MAE: {final_mae}")
print(f"Best Gradient Boosting R2: {final_r2}")


#Visualize Actual vs Predicted Prices

plt.figure(figsize=(8,6))
plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Gradient Boosting)")
plt.show()


#Feature importance for the best Gradient Boosting model

importances = best_gb_model.feature_importances_
features = x_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Gradient Boosting)")
plt.bar(range(x_train.shape[1]), importances[indices], align='center')
plt.xticks(range(x_train.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()


new_data = pd.DataFrame({
    'area':[8500, 9500, 7200],
    'bedrooms':[3, 4, 2],
    'bathrooms':[2, 3, 1],
    'stories':[2, 3, 1],
    'mainroad':['yes','yes','no'],
    'guestroom':['no','yes','no'],
    'basement':['yes','no','no'],
    'hotwaterheating':['no','yes','no'],
    'airconditioning':['yes','yes','no'],
    'parking':[2,3,1],
    'prefarea':['yes','no','no'],
    'furnishingstatus':['furnished','semi-furnished','unfurnished']
})

# Encode categorical columns
for col in new_data.select_dtypes(include=['object']):
    le = LabelEncoder()
    new_data[col] = le.fit_transform(new_data[col])

# Scale features
new_data_scaled = scaler.transform(new_data)

# Predict
pred_prices = best_gb_model.predict(new_data_scaled)
print(pred_prices)



import pickle

# Save the model
with open("best_gb_model.pkl", "wb") as f:
    pickle.dump(best_gb_model, f)

# Later, to load the model
with open("best_gb_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

