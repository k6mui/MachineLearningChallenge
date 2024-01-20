import pandas as pd
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Primero importamos el modelo y las funciones auxiliares
from utils.utils import MATRIZ_COSTE, calcula_prediccion_coste, cargar_y_preparar_datos
from utils.xgboost_model import train_xg


print("Iniciando Proceso de entrenamiento...")

# Path de los datasets de entrenamiento y test
path_train = "../input/data_train.csv"
path_test = "../input/data_test.csv"

print("Cargando los datasets de entrenamiento y test...")
# Separamos el dataset de entrenamiento de las etiquetas y cargamos todo el dataset de test
X_resampled, y_resampled, data_test_df = cargar_y_preparar_datos(path_train, path_test, 54)
print("Carga de datos completada.")

# Entrenamos el modelo
start_time = time.time()
print("Iniciando el entrenamiento del modelo...")
xg = train_xg(X_resampled, y_resampled)
print(f"Entrenamiento completado.\n%Tiempo de entrenamiento ==> {time.time() - start_time:.2f} segundos.")

# Realizamos las predicciones en el conjunto de test
print("Realizando predicciones con el modelo entrenado...")
y_pred_final = calcula_prediccion_coste(xg.predict_proba(data_test_df), MATRIZ_COSTE)
print("Predicciones realizadas.")

# Guardamos las predicciones finales en formato CSV 
predictions_df = pd.DataFrame({'Id': range(len(y_pred_final)), 'Category': y_pred_final})
output_path = 'predicciones_XGBoost.csv'
predictions_df.to_csv(output_path, index=False)
print(f"Predicciones guardadas en {output_path}")


