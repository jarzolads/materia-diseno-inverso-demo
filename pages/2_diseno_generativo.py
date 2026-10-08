import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from pymatgen.core import Composition
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

# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #172B4D;
    }

    [data-testid="stSidebar"],
    [data-testid="stHeader"] {
        background-color: #F5F9FC;
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

    .agent-box {
        background-color: #F8FAFC;
        border: 1px solid #C8D6E5;
        color: #172B4D !important;
        padding: 20px;
        border-radius: 8px;
        line-height: 1.6;
    }

    .agent-box table {
        width: 100% !important;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 14px;
    }

    .agent-box th,
    .agent-box td {
        padding: 6px 8px !important;
        text-align: left;
        vertical-align: top;
        white-space: normal !important;
        word-wrap: break-word;
        border: 1px solid #D5E0E8;
    }

    .agent-box th {
        background-color: #EAF5FB;
        color: #124E78;
    }

    .agent-box td {
        background-color: #FFFFFF;
        color: #172B4D;
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
    '<div class="subtitle">Desarrollado por Dr. Jesús Andrés Arzola Flores</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Esta sección utiliza aprendizaje automático para proponer formulaciones
    hipotéticas a partir de materiales conocidos de Materials Project.

    El modelo aprende relaciones aproximadas entre composición química, densidad
    y energía sobre el envolvente. Después explora nuevas combinaciones químicas
    dentro de diferentes familias de materiales.

    Las formulaciones generadas no son materiales confirmados. Deben validarse
    con cálculos adicionales, revisión bibliográfica y experimentación.
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
# FAMILIAS
# ============================================================

FAMILIAS = {
    "Óxidos metálicos": {
        "sistemas": [
            "Fe-O", "Co-O", "Ni-O", "Mn-O",
            "Ti-O", "Zn-O", "Al-O", "Zr-O"
        ],
        "plantilla": "oxido"
    },

    "Ferritas": {
        "sistemas": [
            "Fe-Co-O", "Fe-Ni-O", "Fe-Mn-O",
            "Fe-Zn-O", "Fe-Cu-O"
        ],
        "plantilla": "ferrita"
    },

    "Aleaciones magnéticas": {
        "sistemas": [
            "Fe-Co", "Fe-Ni", "Co-Ni",
            "Fe-Mn", "Co-Mn"
        ],
        "plantilla": "aleacion"
    },

    "Materiales para baterías": {
        "sistemas": [
            "Li-Fe-O", "Li-Co-O", "Li-Ni-O",
            "Li-Mn-O", "Na-Fe-O", "Na-Mn-O"
        ],
        "plantilla": "bateria"
    },

    "Cerámicos": {
        "sistemas": [
            "Ba-Ti-O", "Sr-Ti-O", "Al-O",
            "Zr-O", "Ca-Ti-O"
        ],
        "plantilla": "ceramico"
    }
}

# ============================================================
# FUNCIONES
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
    try:
        composicion = Composition(formula)
        elementos = composicion.elements

        cantidades = np.array([
            float(composicion[elemento])
            for elemento in elementos
        ])

        fracciones = cantidades / cantidades.sum()

        numeros_atomicos = np.array([
            valor_elemento(elemento, "Z")
            for elemento in elementos
        ])

        masas = np.array([
            valor_elemento(elemento, "atomic_mass")
            for elemento in elementos
        ])

        electronegatividades = np.array([
            valor_elemento(elemento, "X")
            for elemento in elementos
        ])

        radios = np.array([
            valor_elemento(elemento, "atomic_radius")
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


def consultar_dataset_mp(sistemas, maximo):
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

        documentos_totales = []
        ids_obtenidos = set()

        with MPRester(MP_API_KEY) as mpr:

            for sistema in sistemas:

                try:
                    documentos = mpr.materials.summary.search(
                        chemsys=sistema,
                        fields=campos,
                        num_chunks=1
                    )

                    for documento in documentos:

                        material_id = str(
                            getattr(
                                documento,
                                "material_id",
                                ""
                            )
                        )

                        if material_id not in ids_obtenidos:
                            documentos_totales.append(documento)
                            ids_obtenidos.add(material_id)

                        if len(documentos_totales) >= maximo:
                            break

                except Exception:
                    continue

                if len(documentos_totales) >= maximo:
                    break

        datos = preparar_dataset(
            documentos_totales[:maximo]
        )

        return datos, (
            f"Se obtuvieron {len(datos)} materiales de "
            f"{len(sistemas)} sistemas químicos."
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

    modelos = {}

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

    predicciones = modelo_densidad.predict(
        x_test
    )

    modelos["modelo_densidad"] = modelo_densidad
    modelos["mae_densidad"] = mean_absolute_error(
        y_test,
        predicciones
    )
    modelos["r2_densidad"] = r2_score(
        y_test,
        predicciones
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

    predicciones = modelo_estabilidad.predict(
        x_test
    )

    modelos["modelo_estabilidad"] = modelo_estabilidad
    modelos["mae_estabilidad"] = mean_absolute_error(
        y_test,
        predicciones
    )
    modelos["r2_estabilidad"] = r2_score(
        y_test,
        predicciones
    )

    return modelos


def generar_formulaciones(
    familia,
    elemento_a,
    elemento_b,
    elemento_base,
    anion
):
    plantilla = FAMILIAS[
        familia
    ]["plantilla"]

    formulaciones = []

    proporciones = [
        (1.0, 0.0),
        (0.75, 0.25),
        (0.50, 0.50),
        (0.25, 0.75),
        (0.0, 1.0)
    ]

    for fraccion_a, fraccion_b in proporciones:

        try:

            if plantilla == "ferrita":

                partes = []

                if fraccion_a > 0:
                    partes.append(
                        f"{elemento_a}{fraccion_a:g}"
                    )

                if fraccion_b > 0:
                    partes.append(
                        f"{elemento_b}{fraccion_b:g}"
                    )

                formula = (
                    "".join(partes)
                    + f"{elemento_base}2{anion}4"
                )

            elif plantilla == "oxido":

                partes = []

                if fraccion_a > 0:
                    partes.append(
                        f"{elemento_a}{fraccion_a:g}"
                    )

                if fraccion_b > 0:
                    partes.append(
                        f"{elemento_b}{fraccion_b:g}"
                    )

                formula = (
                    "".join(partes)
                    + f"{anion}3"
                )

            elif plantilla == "aleacion":

                partes = []

                if fraccion_a > 0:
                    partes.append(
                        f"{elemento_a}{fraccion_a:g}"
                    )

                if fraccion_b > 0:
                    partes.append(
                        f"{elemento_b}{fraccion_b:g}"
                    )

                formula = "".join(partes)

            elif plantilla == "bateria":

                partes = []

                if fraccion_a > 0:
                    partes.append(
                        f"{elemento_a}{fraccion_a:g}"
                    )

                if fraccion_b > 0:
                    partes.append(
                        f"{elemento_b}{fraccion_b:g}"
                    )

                formula = (
                    "Li"
                    + "".join(partes)
                    + f"{anion}2"
                )

            else:

                partes = []

                if fraccion_a > 0:
                    partes.append(
                        f"{elemento_a}{fraccion_a:g}"
                    )

                if fraccion_b > 0:
                    partes.append(
                        f"{elemento_b}{fraccion_b:g}"
                    )

                formula = (
                    "".join(partes)
                    + f"{elemento_base}{anion}3"
                )

            formula_reducida = Composition(
                formula
            ).reduced_formula

            if formula_reducida not in formulaciones:
                formulaciones.append(
                    formula_reducida
                )

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

        densidad = modelos[
            "modelo_densidad"
        ].predict(vector)[0]

        estabilidad = modelos[
            "modelo_estabilidad"
        ].predict(vector)[0]

        registros.append({
            "Fórmula propuesta": formula,
            "Densidad predicha": densidad,
            "E_hull predicho": estabilidad,
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


def extraer_json(texto):
    texto = texto.replace("```json", "")
    texto = texto.replace("```", "")
    texto = texto.strip()

    inicio = texto.find("{")
    final = texto.rfind("}")

    if inicio == -1 or final == -1:
        raise ValueError(
            "Gemini no devolvió un JSON válido."
        )

    return json.loads(
        texto[inicio:final + 1]
    )


def interpretar_prompt_generativo(prompt):
    valores_defecto = {
        "familia": "Ferritas",
        "elemento_a": "Co",
        "elemento_b": "Ni",
        "elemento_base": "Fe",
        "anion": "O",
        "densidad_objetivo": 5.0
    }

    if not GEMINI_API_KEY:
        return valores_defecto, (
            "Gemini no está conectado. "
            "Se utilizarán valores predeterminados."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        instrucciones = f"""
        Interpreta este objetivo de diseño generativo de materiales:

        {prompt}

        Devuelve exclusivamente un JSON válido con esta estructura:

        {{
          "familia": "Ferritas",
          "elemento_a": "Co",
          "elemento_b": "Ni",
          "elemento_base": "Fe",
          "anion": "O",
          "densidad_objetivo": 5.0
        }}

        La familia debe ser una de estas:

        - Óxidos metálicos
        - Ferritas
        - Aleaciones magnéticas
        - Materiales para baterías
        - Cerámicos

        Si el prompt no contiene un valor, utiliza un valor razonable.
        Utiliza símbolos químicos correctos.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instrucciones
        )

        valores = extraer_json(
            respuesta.text
        )

        for clave, valor in valores_defecto.items():
            if clave not in valores:
                valores[clave] = valor

        if valores["familia"] not in FAMILIAS:
            valores["familia"] = "Ferritas"

        return valores, (
            "Gemini interpretó el prompt generativo."
        )

    except Exception as error:
        return valores_defecto, (
            f"No fue posible interpretar el prompt: {error}"
        )


def interpretar_candidatos_gemini(
    prompt,
    familia,
    resultados
):
    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. Se muestran las predicciones."
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
        Actúa como un experto en diseño computacional de materiales,
        síntesis inorgánica y caracterización experimental.

        Familia:
        {familia}

        Objetivo:
        {prompt}

        Formulaciones propuestas por machine learning:
        {json.dumps(datos, ensure_ascii=False, default=str)}

        Redacta una respuesta clara en español con estas secciones:

        ## 1. Interpretación del objetivo

        Explica qué se intentó diseñar y qué propiedades se priorizaron.

        ## 2. Interpretación de las formulaciones

        Explica cuáles parecen más prometedoras y cuáles son hipotéticas.

        ## 3. Limitaciones del modelo

        Explica que el modelo aprendió relaciones aproximadas a partir de
        materiales conocidos y que las predicciones no son confirmaciones
        experimentales.

        ## 4. Candidato recomendado

        Selecciona una formulación para continuar con cálculos adicionales.

        ## 5. Rutas de síntesis, precursores y caracterización

        Propón una o más rutas de síntesis que podrían explorarse.

        Para cada ruta indica:

        - Método de síntesis.
        - Precursores químicos posibles.
        - Función de cada precursor.
        - Material y equipo de laboratorio.
        - Variables que deberían controlarse.
        - Riesgos principales.
        - Posibles productos secundarios.

        Explica cómo debería caracterizarse el nuevo material.

        Considera, cuando sea pertinente:

        - Difracción de rayos X.
        - SEM o TEM.
        - EDS.
        - FTIR o Raman.
        - VSM o magnetometría.
        - DLS y potencial zeta.
        - Análisis térmico.
        - ICP-OES.

        No inventes cantidades exactas si no existe un protocolo validado.

        ## 6. Reactivos y equipo

        Presenta una tabla compacta:

        | Reactivo o equipo | Función | Observación de seguridad |

        ## 7. Costo preliminar

        Presenta una tabla compacta:

        | Precursor o consumible | Cantidad aproximada | Precio estimado MXN |

        Aclara que los precios deben cotizarse.

        ## 8. Proveedores y SDS

        Sugiere proveedores potenciales como Merck/Sigma-Aldrich,
        Thermo Fisher, Fisher Scientific, Alfa Aesar u otros proveedores
        mexicanos.

        No afirmes disponibilidad actual. Para SDS, proporciona enlaces
        solamente si son oficiales y verificables. Nunca inventes enlaces.

        ## 9. Laboratorios de la BUAP

        Sugiere qué tipos de laboratorios o unidades académicas podrían ser
        adecuados para síntesis y caracterización.

        No inventes nombres, responsables, disponibilidad ni equipos. Indica
        que todo debe confirmarse con la unidad correspondiente.

        ## 10. Siguiente etapa computacional

        Recomienda cálculos estructurales, estabilidad, DFT, relajación
        geométrica o comparación con nuevas bases de datos.

        No presentes ninguna formulación como material confirmado,
        sintetizado, biocompatible o clínicamente seguro.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instrucciones
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible consultar Gemini: {error}"


# ============================================================
# EXPLICACIÓN
# ============================================================

st.markdown(
    '<div class="section">¿Qué hace esta página?</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Esta página entrena modelos de machine learning utilizando datos de varias
    familias de Materials Project. Después genera combinaciones químicas
    hipotéticas dentro de la familia seleccionada.

    El resultado no es todavía un material sintetizado. Es una hipótesis
    computacional que debe continuar con validación estructural, cálculos de
    estabilidad y experimentación.
    """
)


# ============================================================
# MODO DE DEFINICIÓN
# ============================================================

st.markdown(
    '<div class="section">Define la formulación</div>',
    unsafe_allow_html=True
)

modo = st.radio(
    "¿Cómo deseas indicar la formulación?",
    [
        "Elegir los elementos manualmente",
        "Escribir un prompt científico"
    ],
    horizontal=True
)

familia = "Ferritas"
elemento_a = "Co"
elemento_b = "Ni"
elemento_base = "Fe"
anion = "O"
densidad_objetivo = 5.0
prompt_ml = ""

if modo == "Elegir los elementos manualmente":

    st.subheader("Selección manual de elementos")

    st.write(
        """
        Define los elementos que participarán en la formulación.

        **Elemento A y elemento B:** son los elementos que se sustituirán o
        combinarán dentro de la nueva formulación.

        **Elemento base:** es el elemento principal que forma la estructura
        junto con A y B.

        **Anión:** es el elemento que recibe electrones o forma la parte aniónica
        del compuesto. En óxidos y ferritas normalmente es el oxígeno, O.
        """
    )

    familia = st.selectbox(
        "Familia de materiales",
        list(FAMILIAS.keys())
    )

    st.info(
        "Sistemas químicos consultados: "
        + ", ".join(FAMILIAS[familia]["sistemas"])
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        elemento_a = st.text_input(
            "Elemento A",
            value="Co"
        )

    with col2:
        elemento_b = st.text_input(
            "Elemento B",
            value="Ni"
        )

    with col3:
        elemento_base = st.text_input(
            "Elemento base",
            value="Fe"
        )

    with col4:
        anion = st.text_input(
            "Anión",
            value="O"
        )

    prompt_ml = (
        "Diseño definido mediante selección manual de elementos."
    )

else:

    st.subheader("Diseño mediante prompt")

    prompt_ml = st.text_area(
        "Describe el material que deseas diseñar",
        value=(
            "Diseña una ferrita hipotética basada en cobalto, níquel, hierro "
            "y oxígeno, con baja energía sobre el envolvente y densidad "
            "moderada para una posible aplicación magnética."
        ),
        height=150
    )

    st.write(
        """
        Gemini interpretará el prompt y extraerá la familia de materiales, los
        elementos A, B, base y anión, y la densidad objetivo.
        """
    )

    interpretar = st.button(
        "Interpretar prompt generativo",
        use_container_width=True
    )

    if interpretar:

        with st.spinner(
            "Gemini está interpretando la formulación solicitada..."
        ):
            valores, mensaje = interpretar_prompt_generativo(
                prompt_ml
            )

        st.success(mensaje)

        familia = valores["familia"]
        elemento_a = valores["elemento_a"]
        elemento_b = valores["elemento_b"]
        elemento_base = valores["elemento_base"]
        anion = valores["anion"]
        densidad_objetivo = float(
            valores["densidad_objetivo"]
        )

        st.subheader("Elementos interpretados por Gemini")

        criterios = pd.DataFrame([
            {
                "Parámetro": "Familia",
                "Valor": familia
            },
            {
                "Parámetro": "Elemento A",
                "Valor": elemento_a
            },
            {
                "Parámetro": "Elemento B",
                "Valor": elemento_b
            },
            {
                "Parámetro": "Elemento base",
                "Valor": elemento_base
            },
            {
                "Parámetro": "Anión",
                "Valor": anion
            },
            {
                "Parámetro": "Densidad objetivo",
                "Valor": densidad_objetivo
            }
        ])

        st.dataframe(
            criterios,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# OPCIONES DE ENTRENAMIENTO
# ============================================================

st.markdown(
    '<div class="section">Opciones del modelo</div>',
    unsafe_allow_html=True
)

col5, col6 = st.columns(2)

with col5:
    maximo_mp = st.slider(
        "Número máximo de materiales para entrenar",
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
        value=densidad_objetivo,
        step=0.1
    )

ejecutar = st.button(
    "🧠 Generar formulaciones con machine learning",
    type="primary",
    use_container_width=True
)


# ============================================================
# EJECUCIÓN
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

    inicio_total = time.time()

    with st.spinner(
        "Consultando varias familias de Materials Project..."
    ):
        dataset, mensaje = consultar_dataset_mp(
            sistemas=FAMILIAS[familia]["sistemas"],
            maximo=maximo_mp
        )

    st.success(mensaje)

    if dataset.empty:
        st.error(
            "No fue posible construir el conjunto de entrenamiento."
        )
        st.stop()

    with st.spinner(
        "Entrenando los modelos de machine learning..."
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
        '<div class="section">Calidad aproximada del modelo</div>',
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

    formulaciones = generar_formulaciones(
        familia=familia,
        elemento_a=elemento_a.strip(),
        elemento_b=elemento_b.strip(),
        elemento_base=elemento_base.strip(),
        anion=anion.strip()
    )

    formulas_conocidas = set(
        dataset["formula_pretty"].astype(str)
    )

    with st.spinner(
        "Generando y evaluando formulaciones hipotéticas..."
    ):
        resultados = predecir_formulaciones(
            formulaciones=formulaciones,
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
        text="Fórmula propuesta",
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
        "Gemini está preparando las rutas de síntesis, precursores y caracterización..."
    ):
        recomendacion = interpretar_candidatos_gemini(
            prompt=prompt_ml,
            familia=familia,
            resultados=resultados
        )

    st.markdown(
        f"""
        <div class="agent-box">
        {recomendacion}
        </div>
        """,
        unsafe_allow_html=True
    )

    tiempo_total = time.time() - inicio_total

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

    Las rutas de síntesis, precursores, costos, proveedores, SDS y laboratorios
    sugeridos por Gemini deben verificarse antes de realizar cualquier
    experimento.

    Antes de intentar una síntesis se requieren cálculos adicionales, revisión
    bibliográfica, evaluación de seguridad y validación experimental.
    </div>
    """,
    unsafe_allow_html=True
)
