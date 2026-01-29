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
from scipy.stats import pearsonr, ttest_ind, f_oneway

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



def plot_features_num_regression(df: pd.DataFrame, target_col : str = "", columns: list[str] = [], umbral_corr: float = 0.0, pvalue= None):
    """
    Args:
        df (pandas.DataFrame): DataFrame de pandas.
        target_col (str): Nombre de la columna objetivo.
        columns (list[str], opcional): Lista de columnas a analizar.
            Si está vacía, se utilizarán todas las columnas numéricas del dataframe.
        umbral_corr (float, opcional): Umbral mínimo de correlación en valor absoluto.
            Por defecto es 0.
        pvalue (float, opcional): Nivel de significación estadística.
            Si es None, no se aplica test de significación.

    Returns:
        list[str]: Lista de columnas cuya correlación con la columna objetivo
        supera el umbral indicado y, si procede, es estadísticamente significativa.
    """
    columnas_numericas = []
    if len(columns) > 0:
        for columna in columns:
            columna_existe = columna in df.columns
            columna_es_numerica = columna_existe and pd.api.types.is_numeric_dtype(df[columna])

            if columna_es_numerica:
                columnas_numericas.append(columna)
    else:
        columnas_numericas = df.select_dtypes(np.number).drop(columns=target_col).columns

    columnas_elegidas = []

    for col in columnas_numericas:
        tmp = df[[target_col, col]]

        r, p = pearsonr(tmp[target_col], tmp[col])

        if(abs(r) >= umbral_corr):
            if pvalue is None: 
                columnas_elegidas.append(col)
            elif p <= pvalue:
                columnas_elegidas.append(col)

    sns.pairplot(df[columnas_elegidas])
    plt.show()
    
    return columnas_elegidas



# =========================
# VARIABLES CATEGÓRICAS - REGRESIÓN
# =========================

def get_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05):
    """
    Args:
        pd (pandas.DataFrame): Pandas DataFrame.
        target_col (str): name of the target column.
        pvalue (float): Defaults to 0.05
    Returns:
        list[str]: A list of the categorical columns from the dataframe

    La función debe devolver una lista con las columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario hacer (es decir la función debe poder escoger cuál de los dos test que hemos aprendido tiene que hacer).
    """

    if not isinstance(df, pd.DataFrame):
        print("El parametro df debe ser un DataFrame")
        return None
    
    if not isinstance(target_col, str) or target_col.strip() == "":
        print("target_col debe ser un string no vacio")
        return None

    if not target_col in df.columns:
        print("La variable especificada en target_col no existe en el dataframe")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La variable especificada en target_col no es de tipo numerico")
        return None

    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("pvalue debe ser un número entre 0 y 1")
        return None
    
    if df[target_col].nunique() < 10:
        print("El target no es numérico continuo (poca variabilidad)")
        return None
    
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    cat_cols = [col for col in cat_cols if col != target_col]

    selected_features = []

    for col in cat_cols:
        data = df[[col, target_col]].dropna()

        # ignorar columnas sin variabilidad
        if data[col].nunique() < 2:
            continue

        groups = []

        for category in data[col].unique():
            grupo = data[data[col] == category][target_col]
            groups.append(grupo)

        try:
            # 2 grupos → t-test
            if len(groups) == 2:
                stat, p = ttest_ind(groups[0], groups[1], equal_var=False)

            # 3 o más grupos → ANOVA
            else:
                stat, p = f_oneway(*groups)

            if p < pvalue:
                selected_features.append(col)

        except Exception as e:
            print(f"No se pudo evaluar la columna {col}: {e}")

    return selected_features


def plot_features_cat_regression(df, target_col, columns=None, with_individual_plot=False):
    """
    """