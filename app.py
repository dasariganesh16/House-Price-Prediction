import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv('data/Housing.csv')
le=LabelEncoder()

for col in df.select_dtypes(include='object').columns:
     df[col]=le.fit_transform(df[col])
print(df.head())

X=df.drop('price',axis=1)
y=df['price']

print(X.shape)
print(y.shape)
    # Data types

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)

model=LinearRegression()
model.fit(X_train,y_train)


print(model.coef_)
print(model.intercept_)

y_pred = model.predict(X_test)

# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2:", r2_score(y_test, y_pred))
print(y_pred)
pd.set_option('display.float_format', '{:.0f}'.format)
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

# print(comparison.head(10))
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#print("RMSE:", rmse)
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
stories = int(input("Enter stories: "))
mainroad = int(input("Mainroad (1=yes,0=no): "))
guestroom = int(input("Guestroom (1=yes,0=no): "))
basement = int(input("Basement (1=yes,0=no): "))
hotwaterheating = int(input("Hotwater Heating (1=yes,0=no): "))
airconditioning = int(input("Air Conditioning (1=yes,0=no): "))
parking = int(input("Parking spaces: "))
prefarea = int(input("Preferred Area (1=yes,0=no): "))
furnishingstatus = int(input("Furnishing Status (0=furnished,1=semi,2=unfurnished): "))

new_house = pd.DataFrame({
    "area":[area],
    "bedrooms":[bedrooms],
    "bathrooms":[bathrooms],
    "stories":[stories],
    "mainroad":[mainroad],
    "guestroom":[guestroom],
    "basement":[basement],
    "hotwaterheating":[hotwaterheating],
    "airconditioning":[airconditioning],
    "parking":[parking],
    "prefarea":[prefarea],
    "furnishingstatus":[furnishingstatus]
})
predicted_price = model.predict(new_house)

print("Predicted Price:", predicted_price[0])