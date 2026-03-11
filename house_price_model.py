import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
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


#Training the model
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

prediction=model.predict(x_test)


#Evaluating the model
mae=mean_absolute_error(y_test,prediction)
print(f"Mean Absolute Error: {mae}")

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