import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper

def obtener_pipeline_y_datos():
    # 1. Cargar
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        return None, None

    # 2. Limpiar y Renombrar
    nuevos_nombres = {
        'Age': 'Edad', 'Sex': 'Sexo', 'ChestPainType': 'TipoDolorPecho', 
        'RestingBP': 'PresionReposo', 'Cholesterol': 'Colesterol', 
        'FastingBS': 'AzucarAyunas', 'RestingECG': 'ECGReposo', 
        'MaxHR': 'FrecuenciaMax', 'ExerciseAngina': 'AnginaEjercicio', 
        'Oldpeak': 'DepresionST', 'ST_Slope': 'PendienteST', 
        'HeartDisease': 'EnfermedadCardiaca'
    }
    df = df.rename(columns=nuevos_nombres)
    df = df[df['Colesterol'] > 0] 

    X = df.drop('EnfermedadCardiaca', axis=1)
    y = df['EnfermedadCardiaca']

    # 3. Definir variables
    vars_num = ['Edad', 'PresionReposo', 'Colesterol', 'FrecuenciaMax', 'DepresionST']
    vars_cat = ['Sexo', 'TipoDolorPecho', 'ECGReposo', 'AnginaEjercicio', 'AzucarAyunas']
    vars_ord = ['PendienteST']

    # --- SOLUCIÓN AL ERROR: Convertir explícitamente a 'object' (texto) ---
    # Esto asegura que feature-engine no se queje de que no son categóricas
    for col in vars_cat + vars_ord:
        X[col] = X[col].astype(str)  # Forzamos que sean texto (strings)

    # 4. Pipeline
    pipeline = Pipeline([
        # Imputación de numéricas
        ('imputer', SklearnTransformerWrapper(
            transformer=SimpleImputer(strategy='median'), 
            variables=vars_num
        )),
        
        # Outliers en numéricas
        ('winsorizer', Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=vars_num)),
        
        # Codificación (Feature Engine ahora estará feliz porque ya son texto)
        ('one_hot', OneHotEncoder(variables=vars_cat, drop_last=True)),
        ('ordinal', OrdinalEncoder(encoding_method='ordered', variables=vars_ord)),
        
        # Escalado de numéricas
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=vars_num)),
        
        # Modelo
        ('model', RandomForestClassifier(n_estimators=300, min_samples_leaf=4, random_state=42))
    ])

    # 5. Entrenar
    pipeline.fit(X, y)
    
    return pipeline, df