import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Matéria | Diseño inverso de materiales",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILO CLARO
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #172B4D;
    }

    [data-testid="stSidebar"] {
        background-color: #F5F9FC;
    }

    [data-testid="stHeader"] {
        background-color: #FFFFFF;
    }

    .main-title {
        color: #124E78 !important;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #3B5B73 !important;
        font-size: 20px;
        line-height: 1.4;
        margin-bottom: 18px;
    }

    .section-title {
        color: #124E78 !important;
        font-size: 26px;
        font-weight: 700;
        border-bottom: 3px solid #55A6D9;
        padding-bottom: 7px;
        margin-top: 30px;
        margin-bottom: 18px;
    }

    .description-box {
        background-color: #EAF5FB;
        border-left: 6px solid #1976A8;
        color: #172B4D !important;
        padding: 18px;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 16px;
    }

    .concept-box {
        background-color: #F8FAFC;
        border: 1px solid #C8D6E5;
        color: #172B4D !important;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 16px;
    }

    .metric-card {
        background-color: #F1F8FC;
        border: 1px solid #BFD5E4;
        border-top: 5px solid #1976A8;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        min-height: 120px;
    }

    .metric-card h4 {
        color: #34566F !important;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .metric-card h2 {
        color: #124E78 !important;
        font-size: 25px;
    }

    .agent-box {
        background-color: #F8FAFC;
        border: 1px solid #C8D6E5;
        color: #172B4D !important;
        padding: 22px;
        border-radius: 8px;
        line-height: 1.7;
        font-size: 16px;
    }

    .warning-box {
        background-color: #FFF8E6;
        border-left: 6px solid #D89B00;
        color: #553A00 !important;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.6;
    }

    textarea {
        font-size: 16px !important;
        line-height: 1.5 !important;
        background-color: #FFFFFF !important;
        color: #172B4D !important;
    }

    input {
        background-color: #FFFFFF !important;
        color: #172B4D !important;
    }

    label {
        color: #263648 !important;
        font-weight: 600 !important;
    }

    p, li, span, div {
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    '<div class="main-title">Matéria: Agente para el diseño inverso de materiales</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Desarrollado por Jesús Arzola</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description-box">
    <b>¿Para qué sirve Matéria?</b><br><br>

    Matéria es un agente educativo que ayuda a buscar materiales a partir de
    propiedades deseadas. El usuario puede elegir valores numéricos o escribir
    un objetivo científico mediante un prompt.

    <br><br>

    La aplicación consulta materiales inorgánicos sólidos, como óxidos,
    ferritas, cerámicos, aleaciones metálicas, materiales para energía y otros
    compuestos con composición y estructura cristalina.

    <br><br>

    <b>¿Cómo realiza el diseño inverso?</b><br><br>

    En el diseño directo se parte de un material y se calculan sus propiedades.
    En Matéria ocurre lo contrario: el usuario comienza indicando las
    propiedades que desea y el agente busca materiales existentes que se
    aproximan a esos objetivos.

    <br><br>

    Por lo tanto, Matéria realiza una primera etapa de diseño inverso, llamada
    cribado o selección inversa de candidatos. No genera automáticamente una
    composición nueva ni sustituye la validación experimental.
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "La aplicación consulta datos de Materials Project y utiliza Gemini para interpretar los resultados."
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
# CONCEPTOS CIENTÍFICOS
# ============================================================

st.markdown(
    '<div class="section-title">Conceptos científicos</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Estos conceptos aparecen antes de la búsqueda para que una persona que no
    sea especialista pueda comprender qué significa cada propiedad.
    """
)

with st.expander("¿Qué es la magnetización?"):
    st.markdown(
        """
        <div class="concept-box">

        Imagina que un material está formado por muchos pequeños imanes. La
        magnetización describe qué tan alineados están esos pequeños imanes y
        qué tan fuerte responde el material cuando se aplica un campo magnético.

        Una magnetización alta puede ser útil cuando queremos manipular un
        material mediante un imán o mediante un campo magnético alterno.

        En hipertermia magnética puede influir en la respuesta del material ante
        un campo magnético. Sin embargo, una magnetización alta no significa
        automáticamente que el material sea seguro o adecuado para uso
        biomédico.

        También deben estudiarse el tamaño de partícula, la forma, la
        estabilidad en agua, la toxicidad y la biocompatibilidad.

        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("¿Qué es la energía sobre el envolvente?"):
    st.markdown(
        """
        <div class="concept-box">

        La energía sobre el envolvente, llamada <i>energy above hull</i> o
        E<sub>hull</sub>, es una medida calculada por computadora que ayuda a
        estimar qué tan estable puede ser un material.

        Para visualizarlo, imagina una montaña formada por diferentes
        composiciones químicas. La envolvente representa las combinaciones más
        estables. La energía sobre el envolvente indica qué tan lejos se
        encuentra un material de esa región estable.

        Un valor cercano a cero suele ser favorable. Un valor alto puede indicar
        que el material podría transformarse en otras fases más estables.

        Esta propiedad es una guía computacional y no garantiza que el material
        pueda sintetizarse bajo cualquier condición.

        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("¿Qué es la densidad?"):
    st.markdown(
        """
        <div class="concept-box">

        La densidad indica cuánta masa existe dentro de un determinado volumen.

        En nanopartículas y suspensiones, la densidad puede influir en qué tan
        rápido sedimentan las partículas, qué tan fácil pueden separarse
        mediante un campo magnético y cómo se comportan dentro de un líquido.

        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("¿Qué es el ordenamiento magnético?"):
    st.markdown(
        """
        <div class="concept-box">

        El ordenamiento magnético describe cómo se organizan los momentos
        magnéticos de los átomos dentro de un material.

        En un material ferromagnético, muchos momentos magnéticos apuntan en una
        dirección semejante. En un material antiferromagnético, algunos apuntan
        en direcciones opuestas y pueden cancelarse. En un material
        ferrimagnético existe una organización opuesta, pero la cancelación no
        es completa.

        Esta información ayuda a comprender por qué ciertos materiales
        responden fuertemente a un campo magnético.

        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("¿Qué significa diseño inverso en esta aplicación?"):
    st.markdown(
        """
        <div class="concept-box">

        En el diseño directo se elige una composición, por ejemplo Fe3O4, y
        después se calculan sus propiedades.

        En el diseño inverso se realiza el procedimiento contrario:

        <br><br>

        1. El usuario indica las propiedades deseadas.<br>
        2. Gemini interpreta el objetivo si se utilizó un prompt.<br>
        3. Materials Project proporciona materiales existentes.<br>
        4. La aplicación elimina los candidatos que no cumplen los límites.<br>
        5. Gemini compara los mejores candidatos.<br>
        6. Se propone una estrategia preliminar para estudiarlos.

        <br><br>

        En esta aplicación el diseño inverso es un cribado computacional de
        candidatos. Todavía no se generan composiciones nuevas ni se realiza
        una optimización completa mediante aprendizaje automático.

        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("¿Para qué materiales funciona?"):
    st.markdown(
        """
        <div class="concept-box">

        Matéria está orientada principalmente a materiales inorgánicos sólidos
        que pueden describirse mediante una composición química y una estructura
        cristalina.

        Puede utilizarse para explorar:

        <br><br>

        - Óxidos metálicos.<br>
        - Ferritas y materiales magnéticos.<br>
        - Cerámicos inorgánicos.<br>
        - Aleaciones metálicas.<br>
        - Materiales para baterías y energía.<br>
        - Semiconductores.<br>
        - Algunos materiales bidimensionales.

        <br><br>

        No está diseñada directamente para proteínas, polímeros, formulaciones
        farmacéuticas o materiales biológicos. Para esos casos se necesitarían
        otras bases de datos y propiedades específicas.

        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FUNCIONES
# ============================================================

def preparar_datos(datos):
    datos = datos.copy()

    columnas_requeridas = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "density",
        "energy_above_hull",
        "ordering",
        "total_magnetization_normalized_vol",
        "symmetry"
    ]

    for columna in columnas_requeridas:
        if columna not in datos.columns:
            datos[columna] = np.nan

    columnas_numericas = [
        "density",
        "energy_above_hull",
        "total_magnetization_normalized_vol"
    ]

    for columna in columnas_numericas:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    return datos


def extraer_json(texto):
    texto = texto.replace("```json", "")
    texto = texto.replace("```", "")
    texto = texto.strip()

    inicio = texto.find("{")
    final = texto.rfind("}")

    if inicio == -1 or final == -1:
        raise ValueError("Gemini no devolvió un JSON válido.")

    return json.loads(texto[inicio:final + 1])


def interpretar_prompt(prompt):
    valores_defecto = {
        "magnetizacion_minima": 0.0,
        "energia_maxima": 0.10,
        "densidad_maxima": 0.0,
        "elementos": [],
        "sistema_quimico": ""
    }

    if not GEMINI_API_KEY:
        return valores_defecto, (
            "Gemini no está conectado."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        instrucciones = f"""
        Analiza este objetivo de diseño inverso de materiales:

        {prompt}

        Devuelve exclusivamente un JSON válido con esta estructura:

        {{
          "magnetizacion_minima": 0.0,
          "energia_maxima": 0.10,
          "densidad_maxima": 0.0,
          "elementos": [],
          "sistema_quimico": ""
        }}

        Reglas:

        - Si no se menciona magnetización, utiliza 0.
        - Si no se menciona energía, utiliza 0.10.
        - Si no se menciona densidad, utiliza 0.
        - Extrae los elementos mediante sus símbolos químicos.
        - Si aparece un sistema como Fe-O, escríbelo en sistema_quimico.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instrucciones
        )

        valores = extraer_json(respuesta.text)

        for clave, valor in valores_defecto.items():
            if clave not in valores:
                valores[clave] = valor

        return valores, "Gemini interpretó el prompt correctamente."

    except Exception as error:
        return valores_defecto, (
            f"No fue posible interpretar el prompt: {error}"
        )


def consultar_materials_project(
    elementos,
    sistema_quimico,
    limite
):
    if not MP_API_KEY:
        return pd.DataFrame(), (
            "No se encontró MP_API_KEY en Streamlit Secrets."
        )

    try:
        from mp_api.client import MPRester

        campos = [
            "material_id",
            "formula_pretty",
            "chemsys",
            "density",
            "energy_above_hull",
            "ordering",
            "total_magnetization_normalized_vol",
            "symmetry"
        ]

        parametros = {
            "fields": campos,
            "num_chunks": 1
        }

        if sistema_quimico.strip():
            parametros["chemsys"] = sistema_quimico.strip()

        elif elementos:
            parametros["elements"] = elementos

        with MPRester(MP_API_KEY) as mpr:
            try:
                documentos = mpr.materials.summary.search(
                    **parametros
                )

            except Exception:
                campos_sin_magnetizacion = [
                    "material_id",
                    "formula_pretty",
                    "chemsys",
                    "density",
                    "energy_above_hull",
                    "ordering",
                    "symmetry"
                ]

                parametros["fields"] = campos_sin_magnetizacion

                documentos = mpr.materials.summary.search(
                    **parametros
                )

        registros = []

        for documento in documentos[:limite]:
            registros.append({
                "material_id": str(
                    getattr(documento, "material_id", "")
                ),
                "formula_pretty": getattr(
                    documento, "formula_pretty", ""
                ),
                "chemsys": getattr(
                    documento, "chemsys", ""
                ),
                "density": getattr(
                    documento, "density", np.nan
                ),
                "energy_above_hull": getattr(
                    documento, "energy_above_hull", np.nan
                ),
                "ordering": str(
                    getattr(documento, "ordering", "No disponible")
                ),
                "total_magnetization_normalized_vol": getattr(
                    documento,
                    "total_magnetization_normalized_vol",
                    np.nan
                ),
                "symmetry": str(
                    getattr(documento, "symmetry", "No disponible")
                )
            })

        datos = preparar_datos(
            pd.DataFrame(registros)
        )

        return datos, (
            f"Materials Project devolvió {len(datos)} registros."
        )

    except Exception as error:
        return pd.DataFrame(), (
            f"Error al consultar Materials Project: {error}"
        )


def filtrar_materiales(
    datos,
    magnetizacion_minima,
    energia_maxima,
    densidad_maxima
):
    datos = preparar_datos(datos)

    datos = datos[
        datos["energy_above_hull"].fillna(999)
        <= energia_maxima
    ]

    if magnetizacion_minima > 0:
        datos = datos[
            datos["total_magnetization_normalized_vol"].fillna(0)
            >= magnetizacion_minima
        ]

    if densidad_maxima > 0:
        datos = datos[
            datos["density"].fillna(999)
            <= densidad_maxima
        ]

    return datos.sort_values(
        by="energy_above_hull",
        ascending=True
    )


def generar_recomendacion_gemini(prompt, resultados):
    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. No fue posible generar la recomendación."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        candidatos = resultados.head(5).to_dict(
            orient="records"
        )

        instrucciones = f"""
        Actúa como un agente educativo experto en diseño inverso de materiales,
        síntesis de materiales inorgánicos y caracterización experimental.

        Explica todo con un lenguaje claro, como si hablaras con una persona
        que conoce ciencia básica, pero que todavía no domina ciencia de
        materiales.

        Objetivo del usuario:

        {prompt}

        Estos son los cinco mejores candidatos obtenidos de Materials Project:

        {json.dumps(candidatos, ensure_ascii=False, default=str)}

        Explica explícitamente que esto es selección inversa o cribado de
        materiales existentes. No genera automáticamente composiciones nuevas
        y no garantiza que un candidato pueda sintetizarse.

        Redacta la respuesta en español con estas secciones:

        ## 1. Interpretación del objetivo

        Explica qué está buscando el usuario y por qué las propiedades son
        importantes.

        ## 2. Los cinco mejores candidatos

        Presenta una tabla o lista comparativa con fórmula, sistema químico,
        magnetización, energía sobre el envolvente, densidad y ordenamiento
        magnético.

        ## 3. Candidato recomendado

        Selecciona el candidato más interesante y explica sus ventajas,
        limitaciones y compromisos.

        ## 4. Estrategia preliminar de síntesis

        Explica paso a paso una ruta de síntesis razonable. No inventes
        cantidades exactas si no existe un protocolo específico.

        ## 5. Reactivos químicos necesarios

        Incluye precursor, función, pureza recomendable y observaciones de
        seguridad.

        ## 6. Material y equipo de laboratorio

        Incluye matraces, vasos, parrilla, agitación, pH-metro, filtración,
        centrifugación, horno, campana de extracción y equipo de protección.

        ## 7. Costo preliminar

        Estima el costo de los precursores y consumibles en pesos mexicanos.
        Aclara que los precios son aproximados y deben cotizarse.

        ## 8. Proveedores potenciales

        Menciona empresas que normalmente distribuyen reactivos de laboratorio,
        como Merck/Sigma-Aldrich, Fisher Scientific, Thermo Fisher, Alfa Aesar
        o proveedores mexicanos.

        No afirmes que tienen existencia actual. Se debe confirmar disponibilidad,
        presentación, pureza y precio.

        ## 9. Fichas de datos de seguridad

        Indica cómo localizar la SDS oficial de cada precursor. Proporciona un
        enlace únicamente si estás seguro de que es oficial. Nunca inventes
        enlaces. Si no estás seguro, indica que debe buscarse en la página
        oficial del proveedor utilizando el nombre y CAS.

        ## 10. Laboratorios de la BUAP

        Sugiere qué tipo de laboratorios o unidades de la BUAP podrían ser
        adecuados para síntesis y caracterización, como laboratorios de
        materiales, química, física del estado sólido, microscopía, difracción
        de rayos X o magnetometría.

        No inventes nombres, responsables, disponibilidad ni equipos. Si no
        tienes certeza, escribe que debe confirmarse con la unidad académica
        correspondiente de la BUAP.

        ## 11. Caracterización recomendada

        Considera DRX, SEM, TEM, DLS, FTIR, VSM o magnetometría, análisis
        térmico y potencial zeta cuando sean pertinentes.

        ## 12. Limitaciones y seguridad

        Aclara que Materials Project proporciona propiedades calculadas y que la
        síntesis, toxicidad, biocompatibilidad y desempeño biomédico deben
        validarse experimentalmente.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instrucciones
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible generar la recomendación: {error}"


# ============================================================
# MODO DE BÚSQUEDA
# ============================================================

st.markdown(
    '<div class="section-title">Selecciona el modo de búsqueda</div>',
    unsafe_allow_html=True
)

modo_busqueda = st.radio(
    "¿Cómo deseas definir el objetivo?",
    [
        "Elegir valores de propiedades",
        "Escribir un prompt científico"
    ],
    horizontal=True
)

magnetizacion_minima = 0.0
energia_maxima = 0.10
densidad_maxima = 0.0
prompt_usuario = ""
elementos = []
sistema_quimico = ""

# ============================================================
# MODO POR PARÁMETROS
# ============================================================

if modo_busqueda == "Elegir valores de propiedades":

    st.markdown(
        '<div class="section-title">Parámetros de propiedades</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Define los límites de las propiedades que deseas utilizar para filtrar
        los materiales de Materials Project.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        magnetizacion_minima = st.number_input(
            "Magnetización mínima",
            min_value=0.0,
            max_value=2000.0,
            value=300.0,
            step=10.0
        )

    with col2:
        energia_maxima = st.number_input(
            "Energía máxima sobre el envolvente",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            format="%.3f"
        )

    with col3:
        densidad_maxima = st.number_input(
            "Densidad máxima",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            help="Usa 0 para no aplicar este filtro."
        )

    elementos_texto = st.text_input(
        "Elementos químicos opcionales",
        value="Fe, O",
        help="Ejemplo: Fe, O o Li, Co, O"
    )

    sistema_quimico = st.text_input(
        "Sistema químico opcional",
        value="",
        help="Ejemplo: Fe-O. Si se completa, tendrá prioridad."
    )

    elementos = [
        elemento.strip()
        for elemento in elementos_texto.split(",")
        if elemento.strip()
    ]

    prompt_usuario = (
        "Búsqueda definida mediante parámetros numéricos."
    )

# ============================================================
# MODO POR PROMPT
# ============================================================

else:

    st.markdown(
        '<div class="section-title">Objetivo científico</div>',
        unsafe_allow_html=True
    )

    prompt_usuario = st.text_area(
        "Escribe tu prompt científico",
        value=(
            "Busca materiales magnéticos estables basados en hierro, con alta "
            "magnetización, para explorar hipertermia magnética."
        ),
        height=170
    )

    st.write(
        """
        Ejemplo:

        Busca un material estable con alta magnetización, baja densidad y
        posible aplicación en hipertermia magnética. Considera sistemas
        basados en hierro, cobalto, níquel y oxígeno.
        """
    )

# ============================================================
# OPCIONES Y EJECUCIÓN
# ============================================================

st.markdown(
    '<div class="section-title">Consulta a Materials Project</div>',
    unsafe_allow_html=True
)

limite = st.slider(
    "Número máximo de registros que se consultarán",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

buscar = st.button(
    "🔎 Buscar materiales",
    type="primary",
    use_container_width=True
)

if buscar:

    if not MP_API_KEY:
        st.error(
            """
            No se encontró MP_API_KEY.

            En Streamlit Cloud ve a:

            Manage app → Settings → Secrets

            y agrega:

            MP_API_KEY = "tu_clave_de_materials_project"
            """
        )
        st.stop()

    if modo_busqueda == "Escribir un prompt científico":

        inicio_prompt = time.time()

        with st.spinner(
            "Gemini está interpretando el prompt..."
        ):
            valores, mensaje_prompt = interpretar_prompt(
                prompt_usuario
            )

        st.info(mensaje_prompt)

        magnetizacion_minima = float(
            valores.get("magnetizacion_minima", 0.0)
        )

        energia_maxima = float(
            valores.get("energia_maxima", 0.10)
        )

        densidad_maxima = float(
            valores.get("densidad_maxima", 0.0)
        )

        elementos = valores.get(
            "elementos",
            []
        )

        sistema_quimico = valores.get(
            "sistema_quimico",
            ""
        )

        tiempo_prompt = time.time() - inicio_prompt

        st.caption(
            f"Tiempo de interpretación del prompt: "
            f"{tiempo_prompt:.1f} segundos"
        )

        st.subheader("Criterios interpretados por Gemini")

        criterios = pd.DataFrame([
            {
                "Criterio": "Magnetización mínima",
                "Valor": magnetizacion_minima
            },
            {
                "Criterio": "Energía máxima sobre el envolvente",
                "Valor": energia_maxima
            },
            {
                "Criterio": "Densidad máxima",
                "Valor": densidad_maxima
            },
            {
                "Criterio": "Elementos",
                "Valor": ", ".join(elementos)
            },
            {
                "Criterio": "Sistema químico",
                "Valor": sistema_quimico
            }
        ])

        st.dataframe(
            criterios,
            use_container_width=True,
            hide_index=True
        )

    inicio_mp = time.time()

    with st.spinner(
        "Materials Project está buscando materiales..."
    ):
        datos, mensaje_mp = consultar_materials_project(
            elementos=elementos,
            sistema_quimico=sistema_quimico,
            limite=limite
        )

    tiempo_mp = time.time() - inicio_mp

    st.success(mensaje_mp)

    st.caption(
        f"Tiempo de consulta a Materials Project: "
        f"{tiempo_mp:.1f} segundos"
    )

    if datos.empty:
        st.error(
            """
            No se obtuvieron registros. Revisa la clave de Materials Project,
            los elementos y el sistema químico.
            """
        )
        st.stop()

    resultados = filtrar_materiales(
        datos,
        magnetizacion_minima,
        energia_maxima,
        densidad_maxima
    )

    st.markdown(
        '<div class="section-title">Resultados del diseño inverso</div>',
        unsafe_allow_html=True
    )

    if resultados.empty:

        st.warning(
            f"""
            Materials Project devolvió {len(datos)} registros, pero ninguno
            cumple simultáneamente los criterios seleccionados.

            Prueba aumentando la energía máxima, reduciendo la magnetización
            mínima o eliminando el filtro de densidad.
            """
        )

    else:

        mejores = resultados.head(5).copy()

        card1, card2, card3, card4 = st.columns(4)

        with card1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mejores candidatos</h4>
                    <h2>{len(mejores)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with card2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Registros filtrados</h4>
                    <h2>{len(resultados)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with card3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Menor E_hull</h4>
                    <h2>{mejores["energy_above_hull"].min():.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with card4:
            magnetizaciones = mejores[
                "total_magnetization_normalized_vol"
            ].dropna()

            if magnetizaciones.empty:
                valor_magnetizacion = "N/D"
            else:
                valor_magnetizacion = (
                    f"{magnetizaciones.max():.2f}"
                )

            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mayor magnetización</h4>
                    <h2>{valor_magnetizacion}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Cinco mejores candidatos")

        st.dataframe(
            mejores,
            use_container_width=True,
            hide_index=True
        )

        datos_grafica = resultados.dropna(
            subset=[
                "energy_above_hull",
                "total_magnetization_normalized_vol"
            ]
        )

        if not datos_grafica.empty:

            grafica = px.scatter(
                datos_grafica,
                x="energy_above_hull",
                y="total_magnetization_normalized_vol",
                size="density",
                color="chemsys",
                hover_name="formula_pretty",
                hover_data=[
                    "material_id",
                    "density",
                    "ordering"
                ],
                labels={
                    "energy_above_hull":
                        "Energía sobre el envolvente",
                    "total_magnetization_normalized_vol":
                        "Magnetización normalizada",
                    "density": "Densidad",
                    "chemsys": "Sistema químico"
                },
                title="Mapa de candidatos encontrados"
            )

            grafica.update_layout(
                template="plotly_white",
                height=550,
                font=dict(
                    color="#172B4D"
                )
            )

            st.plotly_chart(
                grafica,
                use_container_width=True
            )

        st.markdown(
            '<div class="section-title">Interpretación del agente</div>',
            unsafe_allow_html=True
        )

        with st.spinner(
            "Gemini está analizando los cinco candidatos y preparando "
            "la síntesis, costos, proveedores y caracterización..."
        ):
            inicio_gemini = time.time()

            recomendacion = generar_recomendacion_gemini(
                prompt_usuario,
                mejores
            )

            tiempo_gemini = time.time() - inicio_gemini

        st.caption(
            f"Tiempo de interpretación completa: "
            f"{tiempo_gemini:.1f} segundos"
        )

        st.markdown(
            f"""
            <div class="agent-box">
            {recomendacion}
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# NOTA FINAL
# ============================================================

st.markdown(
    """
    <div class="warning-box">
    <b>Nota importante:</b><br><br>

    Los candidatos provienen de datos computacionales. Las cantidades de
    reactivos, costos, proveedores, fichas SDS y laboratorios sugeridos deben
    verificarse antes de realizar cualquier experimento.

    La disponibilidad de equipos y laboratorios de la BUAP debe confirmarse
    directamente con la unidad académica correspondiente.

    La aplicación realiza cribado computacional y no garantiza síntesis,
    toxicidad, biocompatibilidad ni desempeño clínico.
    </div>
    """,
    unsafe_allow_html=True
)
