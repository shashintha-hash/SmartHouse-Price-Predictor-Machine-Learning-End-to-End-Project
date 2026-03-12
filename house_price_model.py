import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#Loading the dataset
df=pd.read_csv('Housing.csv')
print(df.head())

#Inspecting the dataset
print(df.shape)

print(df.info())


#Handling missing values
print(df.isnull().sum())

#Visualizing the data
plt.hist(df['price'],bins=20)
plt.xlabel('Price')
plt.ylabel('Frequency') 
plt.title('Distribution of House Prices')
#plt.show()

x=df.drop('price',axis=1)
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#Encoding categorical variables
for col in x_train.select_dtypes(include=['object']):
    le=LabelEncoder()
    x_train[col]=le.fit_transform(x_train[col])
    x_test[col]=le.transform(x_test[col])

print(x_train.head())

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

