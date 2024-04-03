#Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from joblib import dump

#Datos

Base = pd.read_csv("Base_Tulum.csv")

#Definir bases

Base2 = Base[['precio',
       'vistaalmar', 'superficie', 'capacidad', 'aire_ac', 'wifi', 'balcon',
       'terraza', 'pueblo', 'tv', 'alberca', 'estacionamiento', 'gimnasio',
       'bar']]

Base_Tulum = Base[['lprecio',
       'vistaalmar', 'superficie', 'capacidad', 'aire_ac', 'wifi', 'balcon',
       'terraza', 'pueblo', 'tv', 'alberca', 'estacionamiento', 'gimnasio',
       'bar']].copy()

X = Base_Tulum[['vistaalmar', 'superficie', 'capacidad', 'aire_ac', 'wifi', 'balcon',
       'terraza', 'pueblo', 'tv', 'alberca', 'estacionamiento', 'gimnasio',
       'bar']]

Y = Base_Tulum['lprecio']

#Conjuntos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=35)

#Precios

test_precio_original = np.exp(y_test)
test_precio_original = test_precio_original.to_frame(name='precio_original')
test_precio_original

#Estandarizar

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
dump(scaler_X,'scaler.joblib')

#Modelo Lineal

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)
y_pred_lineal = modelo_lineal.predict(X_test)
test_precio_estimado_lineal = np.exp(y_pred_lineal)
test_precio_estimado_lineal = pd.DataFrame(test_precio_estimado_lineal, columns=['precio_estimado_lineal'])

#Metricas
mse_lineal = mean_squared_error(test_precio_original, test_precio_estimado_lineal)
mae_lineal = mean_absolute_error(test_precio_original,test_precio_estimado_lineal)
rmse_lineal = np.sqrt(mse_lineal)
r2_lineal = r2_score(test_precio_original,test_precio_estimado_lineal)

dump(modelo_lineal, 'modelo_lineal.joblib')


#Modelo SVR

modelo_SVR = SVR(kernel='rbf', C=1.0, epsilon=0.1)
modelo_SVR.fit(X_train_scaled, y_train)


y_pred_SVR = modelo_SVR.predict(X_test_scaled)

test_precio_estimado_SVR = np.exp(y_pred_SVR)
test_precio_estimado_SVR = pd.DataFrame(test_precio_estimado_SVR, columns=['precio_estimado_SVR'])

# Métricas SVG
mse_SVG = mean_squared_error(test_precio_original, test_precio_estimado_SVR)
mae_SVG = mean_absolute_error(test_precio_original,test_precio_estimado_SVR)
rmse_SVG = np.sqrt(mse_SVG)
r2_SVG = r2_score(test_precio_original,test_precio_estimado_SVR)

dump(modelo_SVR, 'modelo_SVR.joblib')

#Modelo RFR

modelo_RFR = RandomForestRegressor(n_estimators=350, random_state=35,max_depth=7)
modelo_RFR.fit(X_train_scaled, y_train)

y_pred_forest = modelo_RFR.predict(X_test_scaled)
test_precio_estimado_RFR = np.exp(y_pred_forest)
test_precio_estimado_RFR = pd.DataFrame(test_precio_estimado_RFR, columns=['precio_estimado_RFR'])

#Metricas
mse_forest = mean_squared_error(test_precio_original,test_precio_estimado_RFR)
mae_forest = mean_absolute_error(test_precio_original,test_precio_estimado_RFR)
rmse_forest = np.sqrt(mse_forest)
r2_forest = r2_score(test_precio_original,test_precio_estimado_RFR)

dump(modelo_RFR, 'modelo_RFR.joblib')

#Modelo XGB

modelo_xgb = xgb.XGBRegressor(objective='reg:squarederror',colsample_bytree=0.8, learning_rate = 0.03, max_depth = 7, n_estimators = 350, subsample = 0.9 )
modelo_xgb.fit(X_train_scaled, y_train)

y_pred_xgb = modelo_xgb.predict(X_test_scaled)

test_precio_estimado_XGB = np.exp(y_pred_xgb)
test_precio_estimado_XGB = pd.DataFrame(test_precio_estimado_XGB, columns=['precio_estimado_XGB'])

#Metricas

mse_xgb = mean_squared_error(test_precio_original,test_precio_estimado_XGB)
mae_xgb = mean_absolute_error(test_precio_original,test_precio_estimado_XGB)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(test_precio_original,test_precio_estimado_XGB)

dump(modelo_xgb, 'modelo_XGB.joblib')
