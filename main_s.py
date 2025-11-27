import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main_model import obtener_pipeline_y_datos

st.title('HeartSense: Sistema de Soporte para la Detección Temprana de Enfermedades Cardíacas')
st.set_page_config(page_title="Predicción Cardíaca", page_icon="🫀", layout="wide")


st.set_page_config(page_title="Grupo 1", page_icon="📊", layout="wide")

@st.cache_resource
def cargar_todo():
    return obtener_pipeline_y_datos()

pipeline, df = cargar_todo()

if pipeline is None:
    st.error("⚠️ No encuentro 'heart.csv'.")
    st.stop()

st.header("Dashboard de Riesgo Cardíaco")

tab1, tab2 = st.tabs(["🩺 Predicción Individual", "📊 Análisis de Datos"])


with tab1:
    st.subheader("Evaluación de Nuevo Paciente")
    
    with st.form("form_prediccion"):
        st.markdown("### 📝 Ingrese los datos clínicos")
        
        c1, c2, c3 = st.columns(3)
        
        # COLUMNA 1: Datos Básicos
        with c1:
            st.info("Datos del Paciente")
            edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=50, help="Edad del paciente en años cumplidos.")
            
            sexo = st.selectbox(
                "Sexo Biológico", 
                options=["M", "F"], 
                format_func=lambda x: "Masculino" if x == "M" else "Femenino"
            )
            
            presion = st.number_input(
                "Presión Arterial en Reposo (mm Hg)", 
                min_value=80, max_value=220, value=120,
                help="La presión sistólica medida al ingreso (ej: 120)."
            )
            
            colest = st.number_input(
                "Colesterol Sérico (mm/dl)", 
                min_value=80, max_value=600, value=200,
                help="Nivel de colesterol total en sangre."
            )

        with c2:
            st.info("Síntomas")
            
            opciones_dolor = {
                "ASY": "Asintomático (Sin dolor)",
                "NAP": "Dolor No Anginoso (Atípico)",
                "ATA": "Angina Atípica",
                "TA": "Angina Típica (Clásica)"
            }
            dolor = st.selectbox(
                "Tipo de Dolor de Pecho", 
                options=list(opciones_dolor.keys()),
                format_func=lambda x: opciones_dolor[x],
                help="Tipo de molestia reportada por el paciente."
            )
            
            azucar = st.selectbox(
                "¿Glucemia en Ayunas > 120 mg/dl?", 
                options=[0, 1], 
                format_func=lambda x: "Sí (Posible Diabetes)" if x==1 else "No (Normal)",
                help="Indica si el azúcar en sangre es alto."
            )
            
            angina = st.selectbox(
                "¿Siente dolor al hacer ejercicio?", 
                options=["N", "Y"], 
                format_func=lambda x: "Sí" if x=="Y" else "No",
                help="Angina inducida por el esfuerzo físico."
            )

        with c3:
            st.info("Resultados Electrocardiograma")
            
            opciones_ecg = {
                "Normal": "Normal",
                "ST": "Anomalía Onda ST-T",
                "LVH": "Hipertrofia Ventricular"
            }
            ecg = st.selectbox(
                "Electrocardiograma en Reposo", 
                options=list(opciones_ecg.keys()),
                format_func=lambda x: opciones_ecg[x]
            )
            
            frec = st.number_input("Frecuencia Cardíaca Máx.", 60, 220, 150, help="Pulsaciones máximas alcanzadas.")
            
            oldpeak = st.number_input(
                "Depresión del ST (Oldpeak)", 
                min_value=0.0, max_value=6.0, value=0.0, step=0.1,
                help="Valor numérico del descenso del segmento ST."
            )
            
            opciones_slope = {
                "Up": "Ascendente (Normal)",
                "Flat": "Plana (Alerta)",
                "Down": "Descendente (Peligro)"
            }
            slope = st.selectbox(
                "Pendiente del Segmento ST", 
                options=list(opciones_slope.keys()),
                format_func=lambda x: opciones_slope[x]
            )

        st.write("")
        btn_pred = st.form_submit_button("🔍 ANALIZAR RIESGO CARDÍACO", type="primary", use_container_width=True)

    if btn_pred:
        input_data = pd.DataFrame([{
            'Edad': edad, 'Sexo': sexo, 'TipoDolorPecho': dolor, 
            'PresionReposo': presion, 'Colesterol': colest, 
            'AzucarAyunas': azucar, 'ECGReposo': ecg, 
            'FrecuenciaMax': frec, 'AnginaEjercicio': angina, 
            'DepresionST': oldpeak, 'PendienteST': slope
        }])
        
        pred = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0][1]
        
        if pred == 1:
            st.error(f"ALTO RIESGO DETECTADO ({prob:.1%})")
        else:
            st.success(f"BAJO RIESGO ({prob:.1%})")


with tab2:
    st.subheader("Exploración de Datos Históricos")
    
    st.markdown("### 1. Mapa de Calor (Correlaciones)")
    vars_num = ['Edad', 'PresionReposo', 'Colesterol', 'FrecuenciaMax', 'DepresionST']
    
    if st.checkbox("Mostrar Matriz de Correlación", value=True):
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[vars_num].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    st.divider()

    st.markdown("### 2. Distribución de Variables Numéricas")
    var_sel = st.selectbox("Selecciona variable para ver su histograma:", vars_num)
    
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=var_sel, kde=True, hue="EnfermedadCardiaca", palette="husl", ax=ax_hist)
    ax_hist.set_title(f"Distribución de {var_sel} (0=Sano, 1=Enfermo)")
    st.pyplot(fig_hist)

    st.divider()

    st.markdown("### 3. Análisis de Categorías")
    vars_cat = ['Sexo', 'TipoDolorPecho', 'ECGReposo', 'AnginaEjercicio', 'PendienteST']
    cat_sel = st.selectbox("Selecciona variable categórica:", vars_cat)
    
    fig_cat, ax_cat = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x=cat_sel, hue="EnfermedadCardiaca", palette="viridis", ax=ax_cat)
    ax_cat.set_title(f"{cat_sel} vs Enfermedad")
    st.pyplot(fig_cat)