import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from pymatgen.core import Composition, Element
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Matéria | Diseño generativo",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #172B4D;
    }

    .title {
        color: #124E78 !important;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
    }

    .subtitle {
        color: #3B5B73 !important;
        font-size: 20px;
        margin-bottom: 18px;
    }

    .section {
        color: #124E78 !important;
        font-size: 26px;
        font-weight: 700;
        border-bottom: 3px solid #55A6D9;
        padding-bottom: 7px;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    .info-box {
        background-color: #EAF5FB;
        border-left: 6px solid #1976A8;
        color: #172B4D !important;
        padding: 18px;
        border-radius: 8px;
        line-height: 1.6;
    }

    .warning-box {
        background-color: #FFF8E6;
        border-left: 6px solid #D89B00;
        color: #553A00 !important;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.6;
    }

    .metric {
        background-color: #F1F8FC;
        border: 1px solid #BFD5E4;
        border-top: 5px solid #1976A8;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }

    .metric h4 {
        color: #34566F !important;
    }

    .metric h2 {
        color: #124E78 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    '<div class="title">Matéria: Diseño generativo de materiales</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Segunda pestaña | Desarrollado por Jesús Arzola</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Esta sección utiliza aprendizaje automático para proponer formulaciones
    hipotéticas a partir de materiales conocidos.

    El modelo aprende relaciones aproximadas entre composición química,
    densidad y energía sobre el envolvente. Después explora nuevas
    combinaciones químicas y las ordena según los objetivos seleccionados.

    Los candidatos generados no deben considerarse materiales confirmados.
    Deben validarse con cálculos de mayor nivel y posteriormente mediante
    experimentación.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CLAVES
# ============================================================

MP_API_KEY = st.secrets.get(
    "MP_API_KEY",
    os.getenv("MP_API_KEY", "")
)

GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv("GEMINI_API_KEY", "")
)

GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
)

# ============================================================
# FUNCIONES QUÍMICAS
# ============================================================

def valor_elemento(elemento, propiedad, defecto=0.0):
    valor = getattr(elemento, propiedad, None)

    if valor is None:
        return defecto

    try:
        return float(valor)
    except Exception:
        return defecto


def caracteristicas_composicion(formula):
    """
    Convierte una fórmula química en descriptores numéricos.
    """

    try:
        composicion = Composition(formula)

        elementos = composicion.elements
        cantidades = np.array([
            float(composicion[elemento])
            for elemento in elementos
        ])

        fracciones = cantidades / cantidades.sum()

        numeros_atomicos = np.array([
            valor_elemento(elemento, "Z", 0)
            for elemento in elementos
        ])

        masas = np.array([
            valor_elemento(elemento, "atomic_mass", 0)
            for elemento in elementos
        ])

        electronegatividades = np.array([
            valor_elemento(elemento, "X", 0)
            for elemento in elementos
        ])

        radios = np.array([
            valor_elemento(elemento, "atomic_radius", 0)
            for elemento in elementos
        ])

        return [
            len(elementos),
            float(cantidades.sum()),
            float(np.sum(fracciones * numeros_atomicos)),
            float(np.max(numeros_atomicos)),
            float(np.sum(fracciones * masas)),
            float(np.sum(fracciones * electronegatividades)),
            float(np.sum(fracciones * radios)),
            float(np.std(numeros_atomicos))
        ]

    except Exception:
        return [np.nan] * 8


def preparar_dataset(documentos):
    registros = []

    for documento in documentos:
        formula = getattr(
            documento,
            "formula_pretty",
            None
        )

        if not formula:
            continue

        caracteristicas = caracteristicas_composicion(
            formula
        )

        if any(pd.isna(caracteristicas)):
            continue

        registros.append({
            "material_id": str(
                getattr(documento, "material_id", "")
            ),
            "formula_pretty": formula,
            "density": getattr(
                documento,
                "density",
                np.nan
            ),
            "energy_above_hull": getattr(
                documento,
                "energy_above_hull",
                np.nan
            ),
            "features": caracteristicas
        })

    return pd.DataFrame(registros)


def consultar_dataset_mp(maximo):
    if not MP_API_KEY:
        return pd.DataFrame(), (
            "No se encontró MP_API_KEY."
        )

    try:
        from mp_api.client import MPRester

        campos = [
            "material_id",
            "formula_pretty",
            "density",
            "energy_above_hull"
        ]

        with MPRester(MP_API_KEY) as mpr:
            documentos = mpr.materials.summary.search(
                elements=["Fe", "O"],
                fields=campos,
                num_chunks=1
            )

        datos = preparar_dataset(
            documentos[:maximo]
        )

        return datos, (
            f"Se obtuvieron {len(datos)} materiales para entrenar el modelo."
        )

    except Exception as error:
        return pd.DataFrame(), (
            f"No fue posible consultar Materials Project: {error}"
        )


def entrenar_modelos(datos):
    datos = datos.copy()

    datos["density"] = pd.to_numeric(
        datos["density"],
        errors="coerce"
    )

    datos["energy_above_hull"] = pd.to_numeric(
        datos["energy_above_hull"],
        errors="coerce"
    )

    datos = datos.dropna(
        subset=[
            "density",
            "energy_above_hull"
        ]
    )

    if len(datos) < 15:
        raise ValueError(
            "Se necesitan al menos 15 registros válidos."
        )

    matriz_x = np.array(
        datos["features"].tolist()
    )

    resultados = {}

    x_train, x_test, y_train, y_test = train_test_split(
        matriz_x,
        datos["density"],
        test_size=0.20,
        random_state=42
    )

    modelo_densidad = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    modelo_densidad.fit(
        x_train,
        y_train
    )

    predicciones_densidad = modelo_densidad.predict(
        x_test
    )

    resultados["modelo_densidad"] = modelo_densidad
    resultados["mae_densidad"] = mean_absolute_error(
        y_test,
        predicciones_densidad
    )
    resultados["r2_densidad"] = r2_score(
        y_test,
        predicciones_densidad
    )

    x_train, x_test, y_train, y_test = train_test_split(
        matriz_x,
        datos["energy_above_hull"],
        test_size=0.20,
        random_state=42
    )

    modelo_estabilidad = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    modelo_estabilidad.fit(
        x_train,
        y_train
    )

    predicciones_estabilidad = modelo_estabilidad.predict(
        x_test
    )

    resultados["modelo_estabilidad"] = modelo_estabilidad
    resultados["mae_estabilidad"] = mean_absolute_error(
        y_test,
        predicciones_estabilidad
    )
    resultados["r2_estabilidad"] = r2_score(
        y_test,
        predicciones_estabilidad
    )

    return resultados


def generar_formulaciones(
    cation_a,
    cation_b,
    elemento_base,
    elemento_oxigeno
):
    """
    Genera formulaciones hipotéticas de tipo espinela.

    Ejemplo:
    Co0.5Ni0.5Fe2O4

    Estas formulaciones son candidatas hipotéticas y no materiales confirmados.
    """

    formulaciones = []

    composiciones = [
        (1.0, 0.0),
        (0.75, 0.25),
        (0.50, 0.50),
        (0.25, 0.75),
        (0.0, 1.0)
    ]

    for fraccion_a, fraccion_b in composiciones:

        partes = []

        if fraccion_a > 0:
            if fraccion_a == 1.0:
                partes.append(f"{cation_a}")
            else:
                partes.append(f"{cation_a}{fraccion_a:g}")

        if fraccion_b > 0:
            if fraccion_b == 1.0:
                partes.append(f"{cation_b}")
            else:
                partes.append(f"{cation_b}{fraccion_b:g}")

        partes.append(f"{elemento_base}2")
        partes.append(f"{elemento_oxigeno}4")

        formula = "".join(partes)

        try:
            formula_reducida = Composition(
                formula
            ).reduced_formula

            if formula_reducida not in formulaciones:
                formulaciones.append(formula_reducida)

        except Exception:
            continue

    return formulaciones


def predecir_formulaciones(
    formulaciones,
    modelos,
    formulas_conocidas
):
    registros = []

    for formula in formulaciones:
        caracteristicas = caracteristicas_composicion(
            formula
        )

        if any(pd.isna(caracteristicas)):
            continue

        vector = np.array(
            caracteristicas
        ).reshape(1, -1)

        densidad_predicha = modelos[
            "modelo_densidad"
        ].predict(vector)[0]

        estabilidad_predicha = modelos[
            "modelo_estabilidad"
        ].predict(vector)[0]

        registros.append({
            "Formula propuesta": formula,
            "Densidad predicha": densidad_predicha,
            "E_hull predicho": estabilidad_predicha,
            "Estado": (
                "Conocida en el conjunto"
                if formula in formulas_conocidas
                else "Hipotética"
            )
        })

    resultados = pd.DataFrame(registros)

    if resultados.empty:
        return resultados

    return resultados.sort_values(
        by="E_hull predicho",
        ascending=True
    ).reset_index(drop=True)


def interpretar_candidatos_gemini(prompt, resultados):
    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. Se muestran las predicciones del modelo."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        datos = resultados.to_dict(
            orient="records"
        )

        instrucciones = f"""
        Actúa como un experto en diseño computacional de materiales.

        Objetivo del usuario:
        {prompt}

        Formulaciones propuestas por un modelo de machine learning:
        {json.dumps(datos, ensure_ascii=False, default=str)}

        Explica en español:

        1. Qué formulaciones parecen más prometedoras.
        2. Qué significa que sean hipotéticas.
        3. Por qué las predicciones no equivalen a una confirmación experimental.
        4. Qué cálculos adicionales deberían hacerse.
        5. Qué ruta de síntesis podría explorarse.
        6. Qué técnicas de caracterización serían necesarias.

        No presentes las formulaciones como materiales confirmados.
        No inventes protocolos exactos, precios, proveedores ni laboratorios.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instrucciones
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible consultar Gemini: {error}"


# ============================================================
# EXPLICACIÓN DEL MÉTODO
# ============================================================

st.markdown(
    '<div class="section">¿Qué hace esta pestaña?</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Esta pestaña entrena modelos de machine learning con datos conocidos de
    Materials Project. Después genera nuevas combinaciones químicas dentro de
    una familia estructural definida por el usuario.

    En esta demostración se explora una familia tipo espinela con una fórmula
    aproximada:

    AFe₂O₄

    y también formulaciones mixtas como:

    A₀.₅B₀.₅Fe₂O₄

    Las fórmulas generadas son propuestas computacionales. Todavía no se
    conocen necesariamente en la base de datos ni se ha demostrado que puedan
    sintetizarse.
    """
)

# ============================================================
# CONTROLES
# ============================================================

st.markdown(
    '<div class="section">Define la familia química</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    cation_a = st.text_input(
        "Catión A",
        value="Co"
    )

with col2:
    cation_b = st.text_input(
        "Catión B",
        value="Ni"
    )

with col3:
    elemento_base = st.text_input(
        "Elemento base",
        value="Fe"
    )

with col4:
    elemento_oxigeno = st.text_input(
        "Elemento aniónico",
        value="O"
    )

st.markdown(
    '<div class="section">Objetivo del diseño</div>',
    unsafe_allow_html=True
)

prompt_ml = st.text_area(
    "Describe la formulación que deseas explorar",
    value=(
        "Genera una ferrita hipotética con baja energía sobre el envolvente "
        "y densidad moderada para una posible aplicación magnética."
    ),
    height=120
)

col5, col6 = st.columns(2)

with col5:
    maximo_mp = st.slider(
        "Número de materiales para entrenar",
        min_value=20,
        max_value=500,
        value=100,
        step=20
    )

with col6:
    densidad_objetivo = st.number_input(
        "Densidad objetivo aproximada",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

ejecutar = st.button(
    "🧠 Generar formulaciones con machine learning",
    type="primary",
    use_container_width=True
)

# ============================================================
# EJECUCIÓN DEL MODELO
# ============================================================

if ejecutar:

    if not MP_API_KEY:
        st.error(
            """
            No se encontró MP_API_KEY.

            Agrégala en Streamlit Cloud en:

            Manage app → Settings → Secrets
            """
        )
        st.stop()

    if len(cation_a.strip()) == 0:
        st.error("Debes indicar el catión A.")
        st.stop()

    if len(cation_b.strip()) == 0:
        st.error("Debes indicar el catión B.")
        st.stop()

    inicio = time.time()

    with st.spinner(
        "Descargando datos de Materials Project..."
    ):
        dataset, mensaje = consultar_dataset_mp(
            maximo_mp
        )

    st.success(mensaje)

    if dataset.empty:
        st.error(
            "No fue posible construir el conjunto de entrenamiento."
        )
        st.stop()

    with st.spinner(
        "Entrenando modelos de machine learning..."
    ):
        try:
            modelos = entrenar_modelos(
                dataset
            )
        except Exception as error:
            st.error(
                f"No fue posible entrenar los modelos: {error}"
            )
            st.stop()

    st.markdown(
        '<div class="section">Calidad aproximada de los modelos</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric">
                <h4>MAE densidad</h4>
                <h2>{modelos["mae_densidad"]:.3f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric">
                <h4>R² densidad</h4>
                <h2>{modelos["r2_densidad"]:.3f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric">
                <h4>MAE E_hull</h4>
                <h2>{modelos["mae_estabilidad"]:.4f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric">
                <h4>R² E_hull</h4>
                <h2>{modelos["r2_estabilidad"]:.3f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    formulas_nuevas = generar_formulaciones(
        cation_a=cation_a.strip(),
        cation_b=cation_b.strip(),
        elemento_base=elemento_base.strip(),
        elemento_oxigeno=elemento_oxigeno.strip()
    )

    formulas_conocidas = set(
        dataset["formula_pretty"].astype(str)
    )

    with st.spinner(
        "Generando y evaluando formulaciones hipotéticas..."
    ):
        resultados = predecir_formulaciones(
            formulaciones=formulas_nuevas,
            modelos=modelos,
            formulas_conocidas=formulas_conocidas
        )

    st.markdown(
        '<div class="section">Formulaciones propuestas</div>',
        unsafe_allow_html=True
    )

    if resultados.empty:
        st.warning(
            "No fue posible generar formulaciones válidas."
        )
        st.stop()

    resultados["Diferencia respecto a densidad objetivo"] = abs(
        resultados["Densidad predicha"]
        - densidad_objetivo
    )

    resultados = resultados.sort_values(
        by=[
            "E_hull predicho",
            "Diferencia respecto a densidad objetivo"
        ],
        ascending=[
            True,
            True
        ]
    ).reset_index(drop=True)

    resultados.insert(
        0,
        "Ranking",
        range(1, len(resultados) + 1)
    )

    st.dataframe(
        resultados,
        use_container_width=True,
        hide_index=True
    )

    grafica = px.scatter(
        resultados,
        x="Densidad predicha",
        y="E_hull predicho",
        color="Estado",
        text="Formula propuesta",
        hover_data=[
            "Ranking",
            "Densidad predicha",
            "E_hull predicho"
        ],
        title="Mapa de formulaciones generadas"
    )

    grafica.update_traces(
        textposition="top center"
    )

    grafica.update_layout(
        template="plotly_white",
        height=500,
        font=dict(
            color="#172B4D"
        )
    )

    st.plotly_chart(
        grafica,
        use_container_width=True
    )

    st.markdown(
        '<div class="section">Interpretación del agente</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Gemini está interpretando las nuevas formulaciones..."
    ):
        recomendacion = interpretar_candidatos_gemini(
            prompt_ml,
            resultados
        )

    st.markdown(
        f"""
        <div class="agent-box">
        {recomendacion}
        </div>
        """,
        unsafe_allow_html=True
    )

    tiempo_total = time.time() - inicio

    st.caption(
        f"Tiempo total del experimento computacional: "
        f"{tiempo_total:.1f} segundos"
    )

# ============================================================
# ADVERTENCIA
# ============================================================

st.markdown(
    """
    <div class="warning-box">
    <b>Advertencia científica:</b><br><br>

    Las formulaciones generadas son hipótesis computacionales. El modelo aprende
    patrones a partir de materiales conocidos, pero no demuestra por sí mismo
    estabilidad cristalina, sintetizabilidad, toxicidad o desempeño.

    Antes de intentar una síntesis se requieren cálculos adicionales, revisión
    bibliográfica, evaluación de seguridad y validación experimental.
    </div>
    """,
    unsafe_allow_html=True
)
