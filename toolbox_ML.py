"""
Funciones para análisis exploratorio y selección de features.

Autores: 
María Lucía Jiménez Padilla
Kelly Escalante
María Jesús Sánchez Pimienta
Mario Edgardo López Valladares
Enrique Carreras Franco
"""

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, pearsonr
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder


# =========================
# FUNCIONES GENERALES
# =========================

def describe_df(df):
    """
    Devuelve un DataFrame resumen con tipo de dato, % de nulos,
    número de valores únicos y % de cadrinalidad.
    
    Argumentos:
    df : pandas.DataFrame

    Retorna:
    pandas.DataFrame: DataFrame con columnas:
        - tipo de dato (data_type)
        - % de nulos (missing_%)
        - nº de valores únicos (unique_values)
        - % de cardinalidad (cardin_%)
    """

    summary = pd.DataFrame(index=["data_type", "missing_%", "unique_values", "cardin_%"])

    for col in df.columns:
        summary[col] = [
            df[col].dtype,
            df[col].isnull().mean() * 100,
            df[col].nunique(),
            df[col].nunique() / len(df) * 100
        ]
    return summary


def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere el tipo de cada variable del DataFrame según su cardinalidad
    y porcentaje de cardinalidad.

    Parameters
    ----------
    df : pandas.DataFrame
    umbral_categoria : int
    umbral_continua : float

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas:
        - nombre_variable
        - tipo_sugerido
    """

    resultados = []

    n_filas = len(df)

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = cardinalidad / n_filas

        if cardinalidad == 2:
            tipo = "Binaria"

        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"

        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"

        resultados.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultados)



# =========================
# VARIABLES NUMÉRICAS - REGRESIÓN
# =========================

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve una lista de variables numéricas cuya correlación con el target
    supera un umbral dado y opcionalmente pasa un test de hipótesis.

    Argumentos:
    df (pandas.DataFrame): dataframe
    target_col (str): nombre de la columna target (debe ser una 
        variable numérica continua o discreta pero con alta cardinalidad)
    umbral_corr (float): umbral de correlación entre 0 y 1
    pvalue (float o None): por defecto None

    Retorna:
    list o None
    """


    # checks de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas DataFrame")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no es una columna del DataFrame")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser una variable numérica")
        return None

    # Comprobación de alta cardinalidad (numérica continua)
    cardinalidad = df[target_col].nunique()
    porcentaje_cardinalidad = cardinalidad / len(df)

    if porcentaje_cardinalidad < 0.05:
        print("Error: target_col no parece ser una variable numérica continua")
        return None

    if not isinstance(umbral_corr, float) or not (0 < umbral_corr < 1):
        print("Error: umbral_corr debe ser un float entre 0 y 1")
        return None

    if pvalue is not None:
        if not isinstance(pvalue, float) or not (0 < pvalue < 1):
            print("Error: pvalue debe ser None o un float entre 0 y 1")
            return None

    # SELECCIÓN DE FEATURES
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols.remove(target_col)

    if len(num_cols) == 0:
        print("Error: no hay variables numéricas para evaluar")
        return None

    selected_features = []

    for col in num_cols:
        # Eliminar nulos de forma segura
        valid_data = df[[col, target_col]].dropna()

        if valid_data.shape[0] < 2:
            continue

        corr, p_corr = pearsonr(valid_data[col], valid_data[target_col])

        if abs(corr) > umbral_corr:
            if pvalue is None:
                selected_features.append(col)
            else:
                if p_corr <= pvalue:
                    selected_features.append(col)

    return selected_features



def plot_features_num_regression(df, target_col, columns=None):
    """
    """



# =========================
# VARIABLES CATEGÓRICAS - REGRESIÓN
# =========================

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    """



def plot_features_cat_regression(df, target_col, columns=None, with_individual_plot=False):
    """
    """




