import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuración de la página
st.set_page_config(
    page_title="Aplicación de Predicciones IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Menú de selección
st.sidebar.title("🔍 Menú de Predicción")
opciones = ["Predicción Notas Finales", "Predicción Uso del Teléfono"]
seleccion = st.sidebar.radio("Selecciona la funcionalidad:", opciones)

if seleccion == "Predicción Notas Finales":
    # --- Módulo: Predicción de Nota del Examen Final ---
    st.title("📊 Predicción de Nota del Examen Final")

    # Rutas de archivos
    MODEL_PATH  = "models/modelo_notas.h5"
    SCALER_PATH = "scaler/scaler_notas.pkl"
    DATA_PATH   = "data/notas_estudiantes.csv"

    @st.cache_data
    def load_model_and_scaler(model_path, scaler_path):
        model  = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler

    @st.cache_data
    def load_dataset(path):
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        df.columns = df.columns.str.strip().str.capitalize()
        df = df.rename(columns={'Actividad': 'A', 'Ef': 'EF'})
        return df

    try:
        model_nf, scaler_nf = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        df_nf = load_dataset(DATA_PATH)
    except Exception as e:
        st.error(f"Error al cargar recursos: {e}")
        st.stop()

    X_nf = df_nf[["Parciales", "A"]]
    y_nf = df_nf["EF"]

    # Métricas
    preds_nf = model_nf.predict(scaler_nf.transform(X_nf))
    mae_nf   = mean_absolute_error(y_nf, preds_nf)

    st.markdown(
        f"> **MAE del modelo:** {mae_nf:.3f}  •  **Intervalo 95%:** ±{1.96 * mae_nf:.3f}"
    )

    # Entradas de usuario
    st.sidebar.header("🔧 Parámetros Examen Final")
    parciales = st.sidebar.number_input("Puntaje en Parciales", 0, 30, 26)
    a_val     = st.sidebar.number_input("Puntaje en Actividades", 0, 30, 24)
    if st.sidebar.button("▶️ Predecir Nota Final"):
        X_new = scaler_nf.transform([[parciales, a_val]])
        ef_pred = float(model_nf.predict(X_new)[0][0])
        lower   = ef_pred - 1.96 * mae_nf
        upper   = ef_pred + 1.96 * mae_nf

        col1, col2 = st.columns(2)
        col1.metric(label="Nota Final Predicha", value=f"{ef_pred:.1f}")
        col2.metric(label="Intervalo 95%", value=f"[{lower:.1f}, {upper:.1f}]")

    with st.expander("📋 Ver datos de entrenamiento"): 
        st.dataframe(df_nf, use_container_width=True)

else:
    # --- Módulo: Predicción Uso del Teléfono ---
    st.title("📱 Predicción de Horas de Uso del Teléfono")

    # Ruta fija al CSV
    data_path = 'data/form_uso_telefono.csv'
    try:
        df = pd.read_csv(data_path, encoding='latin1')
    except FileNotFoundError:
        st.error(f"No se encontró el archivo en '{data_path}'.")
        st.stop()

    # Preprocesamiento
    df = df.drop(columns=['Marca temporal', 'Columna 6'], errors='ignore')
    df.columns = [
        'redes_sociales', 'tiene_juegos', 'num_juegos',
        'freq_academicas', 'horas_texto',
        'freq_canvas', 'freq_email'
    ]
    df['tiene_juegos'] = df['tiene_juegos'].map({'Si': 1, 'No': 0})
    df['num_juegos'] = df['num_juegos'].fillna(0).astype(int)
    df['horas_texto'] = df['horas_texto'].str.replace(r'[^0-9:]', '', regex=True)
    df['horas_dia'] = pd.to_timedelta(df['horas_texto']).dt.total_seconds() / 3600

    # Features y target
    X = df[[
        'redes_sociales', 'tiene_juegos', 'num_juegos',
        'freq_academicas', 'freq_canvas', 'freq_email'
    ]]
    y = df['horas_dia']

    # Verificar nulos
    if X.isnull().any().any() or y.isnull().any():
        st.error("El dataset contiene valores faltantes (NaN). Revisa la limpieza de datos.")
        st.stop()

    # --- K-Fold Cross Validation ---
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_tel = LinearRegression()

    # Evaluación con K-Fold
    mse_scores = cross_val_score(model_tel, X, y, scoring='neg_mean_squared_error', cv=kf)
    r2_scores = cross_val_score(model_tel, X, y, scoring='r2', cv=kf)

    # Entrenamiento final con todos los datos
    model_tel.fit(X, y)

    st.subheader("📊 Métricas del Modelo con K-Fold (5 pliegues)")
    st.write(f"**MSE promedio:** {(-mse_scores.mean()):.4f}")
    st.write(f"**R² promedio:** {r2_scores.mean():.4f}")

    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coeficiente': model_tel.coef_
    })
    st.write("**Coeficientes de la regresión:**")
    st.dataframe(coef_df)

    st.markdown("---")
    st.subheader("🔮 Predicción")

    redes = st.number_input("¿Cuántas redes sociales utilizas regularmente?", 0, 10, 3)
    tiene_juegos_str = st.selectbox("¿Tienes juegos instalados en tú télefono?", ("Si", "No"))
    num_juegos = 0
    if tiene_juegos_str == "Si":
        num_juegos = st.number_input("¿Cuántos juegos tienes instalados?", 0, 20, 1)
    freq_acad = st.slider("¿Con qué frecuencia usas el teléfono para actividades académicas?", 1, 5, 3)
    freq_canvas = st.slider("¿Con qué frecuencia utilizas tu teléfono para acceder a la plataforma Canvas?", 1, 5, 3)
    freq_email = st.slider("¿Con qué frecuencia utilizas tu teléfono para revisar tu correo electrónico institucional o personal?", 1, 5, 3)

    if st.button("Predecir horas de uso"):
        tiene_juegos_val = 1 if tiene_juegos_str == "Si" else 0
        X_new = np.array([[redes, tiene_juegos_val, num_juegos, freq_acad, freq_canvas, freq_email]])
        horas_pred = model_tel.predict(X_new)[0]
        st.success(f"Predicción: **{horas_pred:.2f}** horas al día")
