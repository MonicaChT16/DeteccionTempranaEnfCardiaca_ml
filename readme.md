# 🫀 Detección Temprana de Riesgo Cardíaco (Machine Learning)

Este proyecto es una aplicación web interactiva desarrollada con **Python** y **Streamlit** que utiliza algoritmos de Machine Learning para estimar la probabilidad de que un paciente sufra una enfermedad cardíaca.

El sistema funciona como una herramienta de soporte a la decisión clínica, analizando 11 variables fisiológicas y síntomas para predecir el riesgo en tiempo real.

## 🚀 Características

- **Modelo Predictivo:** Utiliza un **Random Forest** optimizado (Recall > 94%).
- **Pipeline Robusto:** Incluye imputación de datos, manejo de outliers, codificación de variables categóricas y escalado numérico.
- **Balanceo de Clases:** Implementación de técnicas para manejar el desbalance de datos.
- **Interfaz Amigable:** Formulario web intuitivo para ingreso de datos médicos.
- **Gráficos Interactivos:** Visualización de datos históricos y correlaciones.


## 📦 Instalación y Configuración

Sigue estos pasos para levantar el proyecto en tu computadora:

### Crear un Entorno Virtual (Recomendado)

Es una buena práctica para aislar las librerías del proyecto.

**En Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**En Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

###  Instalar Dependencias

Instala todas las librerías necesarias (Streamlit, Scikit-learn, etc.) con un solo comando:

```bash
pip install (...)
```

*(Nota: Asegúrate de que el archivo `heart.csv` se encuentre en la carpeta principal del proyecto).*

## ▶️ Ejecución

Una vez instaladas las dependencias, inicia la aplicación con:

```bash
streamlit run main_s.py
```

La aplicación se abrirá automáticamente en tu navegador en la dirección: `http://localhost:8501`.


## 🧠 Tecnologías Utilizadas

  * **Python**
  * **Streamlit** (Frontend)
  * **Scikit-Learn** (Modelado)
  * **Feature-Engine** (Preprocesamiento avanzado)
  * **Imbalanced-Learn** (SMOTE para balanceo)
  * **Pandas & Numpy** (Manipulación de datos)
  * **Matplotlib & Seaborn** (Visualización)

