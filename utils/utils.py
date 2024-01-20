import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Matriz de coste usada en el reto
MATRIZ_COSTE = [
    [0, 5, 1, 1, 1, 1, 1],
    [10, 0, 1, 1, 1, 1, 1],
    [20, 20, 0, 5, 5, 50, 5],
    [20, 20, 10, 0, 1, 50, 5],
    [20, 100, 5, 1, 0, 5, 5],
    [5, 10, 10, 5, 1, 0, 1],
    [10, 5, 1, 1, 1, 1, 0]
]

# Función para realizar el ajuste de las predicciones en base al coste asociado
def calcula_prediccion_coste(array_prob, matriz_coste):
    min_cost_clase = np.matmul(array_prob, matriz_coste)
    return np.argmin(min_cost_clase, axis=1)

def cargar_y_preparar_datos(path_train, path_test, target_column_index):
    # Cargamos los datos
    data_train_df = pd.read_csv(path_train)
    data_test_df = pd.read_csv(path_test)

    # Separamos características y etiquetas
    X_data = data_train_df.drop(data_train_df.columns[target_column_index], axis=1)
    Y_data = data_train_df.iloc[:, target_column_index]

    # Aplicamos SMOTE con kvecinos = 3
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_data, Y_data)

    return X_resampled, y_resampled, data_test_df
