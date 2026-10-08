import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Matéria | Diseño inverso de materiales",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILOS DE LA INTERFAZ
# ============================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }

    .main-title {
        color: #123B67;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #4A5568;
        font-size: 20px;
        line-height: 1.4;
        margin-bottom: 18px;
    }

    .description-box {
        background-color: #EAF3FA;
        border-left: 6px solid #1D70A2;
        color: #172B4D;
        padding: 18px;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 16px;
    }

    .section-title {
        color: #123B67;
        font-size: 26px;
        font-weight: 700;
        border-bottom: 3px solid #3D9BC3;
        padding-bottom: 7px;
        margin-top: 30px;
        margin-bottom: 18px;
    }

    .concept-box {
        background-color: #F7FAFC;
        border: 1px solid #CBD5E0;
        color: #1A202C;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.6;
    }

    .metric-card {
        background-color: #F1F7FB;
        border: 1px solid #C7DCEB;
        border-top: 5px solid #1D70A2;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        min-height: 120px;
    }

    .metric-card h4 {
        color: #40566D;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .metric-card h2 {
        color: #123B67;
        font-size: 25px;
    }

    .agent-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E0;
        color: #1A202C;
        padding: 20px;
        border-radius: 8px;
        line-height: 1.65;
    }

    .warning-box {
        background-color: #FFF7E6;
        border-left: 6px solid #D99000;
        color: #5A3A00;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.6;
    }

    textarea {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    label {
        font-weight: 600 !important;
        color: #263648 !important;
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
    Matéria es un agente de inteligencia artificial para buscar materiales a
    partir de las propiedades que se desean obtener. El usuario puede elegir
    valores numéricos o escribir un objetivo científico mediante un prompt.
    Después, el agente consulta Materials Project, filtra los candidatos y
    genera una interpretación científica preliminar.
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Los resultados se obtienen mediante consultas a Materials Project."
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

with st.expander("Magnetización"):
    st.markdown(
        """
        <div class="concept-box">
        La magnetización es el momento magnético por unidad de volumen. Indica
        qué tan intensamente responde un material frente a un campo magnético
        externo.

        En hipertermia magnética puede relacionarse con la respuesta ante un
        campo alterno. Sin embargo, también deben estudiarse el tamaño de
        partícula, la estabilidad coloidal, la toxicidad y la biocompatibilidad.
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("Energía sobre el envolvente"):
    st.markdown(
        """
        <div class="concept-box">
        La energía sobre el envolvente, conocida como <i>energy above hull</i>
        o E<sub>hull</sub>, es un indicador computacional de estabilidad
        termodinámica.

        Un valor cercano a cero indica que el material se encuentra cerca de la
        envolvente de estabilidad. No garantiza por sí solo que pueda
        sintetizarse experimentalmente.
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("Densidad"):
    st.markdown(
        """
        <div class="concept-box">
        La densidad es la masa por unidad de volumen. Puede influir en la
        sedimentación, la separación magnética y la preparación de suspensiones.
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("Diseño inverso"):
    st.markdown(
        """
        <div class="concept-box">
        En el diseño directo se parte de una composición y se calculan sus
        propiedades.

        En el diseño inverso se parte de las propiedades deseadas y se buscan
        composiciones que potencialmente puedan cumplirlas.

        <br><br>
        <b>Flujo del agente:</b>
        <br>
        1. Definir propiedades o escribir un objetivo.<br>
        2. Consultar Materials Project.<br>
        3. Filtrar los materiales.<br>
        4. Comparar los candidatos.<br>
        5. Interpretar los resultados con Gemini.
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
        raise ValueError("Gemini no devolvió JSON válido.")

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
            "Gemini no está conectado. Utiliza la búsqueda por parámetros."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        instrucciones = f"""
        Analiza este objetivo de diseño inverso de materiales:

        {prompt}

        Devuelve exclusivamente JSON válido con esta estructura:

        {{
          "magnetizacion_minima": 0.0,
          "energia_maxima": 0.10,
          "densidad_maxima": 0.0,
          "elementos": [],
          "sistema_quimico": ""
        }}

        Reglas:

        - Si no se menciona magnetización, usa 0.
        - Si no se menciona E_hull, usa 0.10.
        - Si no se menciona densidad, usa 0.
        - Si se mencionan elementos, usa sus símbolos químicos.
        - Si se menciona un sistema como Fe-O, escríbelo en sistema_quimico.
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


def interpretar_resultados(prompt, resultados):
    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. Se muestran los resultados obtenidos "
            "directamente desde Materials Project."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        registros = resultados.head(20).to_dict(
            orient="records"
        )

        solicitud = f"""
        Interpreta estos resultados de Materials Project.

        Objetivo del usuario:
        {prompt}

        Candidatos:
        {json.dumps(registros, ensure_ascii=False, default=str)}

        Responde en español con:

        1. Interpretación del objetivo.
        2. Candidatos más interesantes.
        3. Compromisos entre las propiedades.
        4. Limitaciones de los datos.
        5. Caracterización experimental recomendada.
        6. Estrategia preliminar de síntesis.

        No afirmes que un material es automáticamente biocompatible o clínico.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=solicitud
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible interpretar los resultados: {error}"


# ============================================================
# MODO DE BÚSQUEDA
# ============================================================

st.markdown(
    '<div class="section-title">Selecciona el modo de búsqueda</div>',
    unsafe_allow_html=True
)

modo_busqueda = st.radio(
    "¿Cómo deseas definir el material?",
    [
        "Elegir valores de propiedades",
        "Escribir un prompt científico"
    ],
    horizontal=True
)

# ============================================================
# VARIABLES INICIALES
# ============================================================

magnetizacion_minima = 0.0
energia_maxima = 0.10
densidad_maxima = 0.0
prompt_usuario = ""
elementos = []
sistema_quimico = ""

# ============================================================
# BÚSQUEDA POR PARÁMETROS
# ============================================================

if modo_busqueda == "Elegir valores de propiedades":

    st.markdown(
        '<div class="section-title">Define las propiedades objetivo</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Selecciona los límites de las propiedades que deseas utilizar para
        filtrar los materiales de Materials Project.
        """
    )

    columna_1, columna_2, columna_3 = st.columns(3)

    with columna_1:
        magnetizacion_minima = st.number_input(
            "Magnetización mínima",
            min_value=0.0,
            max_value=2000.0,
            value=300.0,
            step=10.0
        )

    with columna_2:
        energia_maxima = st.number_input(
            "Energía máxima sobre el envolvente",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            format="%.3f"
        )

    with columna_3:
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
        help="Ejemplo: Fe-O. Si se completa, tiene prioridad."
    )

    elementos = [
        elemento.strip()
        for elemento in elementos_texto.split(",")
        if elemento.strip()
    ]

    prompt_usuario = (
        "Búsqueda mediante parámetros numéricos de propiedades."
    )

# ============================================================
# BÚSQUEDA POR PROMPT
# ============================================================

else:

    st.markdown(
        '<div class="section-title">Describe tu objetivo científico</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Escribe el tipo de material o aplicación que deseas explorar. Gemini
        convertirá tu solicitud en criterios de búsqueda.
        """
    )

    prompt_usuario = st.text_area(
        "Prompt científico",
        value=(
            "Busca materiales magnéticos estables basados en hierro, con alta "
            "magnetización, para explorar hipertermia magnética."
        ),
        height=160,
        help=(
            "Puedes mencionar aplicaciones, elementos, magnetización, "
            "estabilidad o densidad."
        )
    )

    ejecutar_interpretacion = st.button(
        "Interpretar prompt con Gemini",
        use_container_width=True
    )

    if ejecutar_interpretacion:

        with st.spinner("Gemini está interpretando el objetivo..."):
            valores, mensaje = interpretar_prompt(
                prompt_usuario
            )

        st.success(mensaje)

        magnetizacion_minima = float(
            valores.get("magnetizacion_minima", 0.0)
        )

        energia_maxima = float(
            valores.get("energia_maxima", 0.10)
        )

        densidad_maxima = float(
            valores.get("densidad_maxima", 0.0)
        )

        elementos = valores.get("elementos", [])
        sistema_quimico = valores.get(
            "sistema_quimico",
            ""
        )

        st.subheader("Parámetros interpretados")

        tabla_parametros = pd.DataFrame([
            {
                "Propiedad": "Magnetización mínima",
                "Valor": magnetizacion_minima
            },
            {
                "Propiedad": "Energía máxima sobre el envolvente",
                "Valor": energia_maxima
            },
            {
                "Propiedad": "Densidad máxima",
                "Valor": densidad_maxima
            },
            {
                "Propiedad": "Elementos",
                "Valor": ", ".join(elementos)
            },
            {
                "Propiedad": "Sistema químico",
                "Valor": sistema_quimico
            }
        ])

        st.dataframe(
            tabla_parametros,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# OPCIONES GENERALES
# ============================================================

st.markdown(
    '<div class="section-title">Opciones de consulta</div>',
    unsafe_allow_html=True
)

limite_resultados = st.slider(
    "Número máximo de registros que se solicitarán",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

buscar = st.button(
    "🔎 Buscar materiales en Materials Project",
    type="primary",
    use_container_width=True
)

# ============================================================
# CONSULTA
# ============================================================

if buscar:

    if not MP_API_KEY:
        st.error(
            """
            No se encontró `MP_API_KEY`.

            En Streamlit Cloud agrega esta clave en:

            `Manage app → Settings → Secrets`

            Ejemplo:

            `MP_API_KEY = "tu_clave_de_materials_project"`
            """
        )
        st.stop()

    with st.spinner(
        "Consultando directamente Materials Project..."
    ):
        datos, mensaje = consultar_materials_project(
            elementos=elementos,
            sistema_quimico=sistema_quimico,
            limite=limite_resultados
        )

    st.success(mensaje)

    if datos.empty:
        st.error(
            """
            No se obtuvieron resultados. Revisa la clave de Materials Project,
            los símbolos químicos y el sistema químico.
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
            cumple simultáneamente con los límites seleccionados.

            Prueba aumentando la energía máxima, reduciendo la magnetización
            mínima o eliminando el filtro de densidad.
            """
        )

    else:

        tarjeta_1, tarjeta_2, tarjeta_3, tarjeta_4 = st.columns(4)

        with tarjeta_1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Candidatos</h4>
                    <h2>{len(resultados)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tarjeta_2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Sistemas químicos</h4>
                    <h2>{resultados["chemsys"].nunique()}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tarjeta_3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Menor E_hull</h4>
                    <h2>{resultados["energy_above_hull"].min():.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tarjeta_4:
            magnetizaciones = resultados[
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

        st.dataframe(
            resultados,
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
                    "density":
                        "Densidad",
                    "chemsys":
                        "Sistema químico"
                },
                title="Mapa de materiales encontrados"
            )

            grafica.update_layout(
                template="plotly_white",
                height=550
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
            "Gemini está interpretando los candidatos..."
        ):
            interpretacion = interpretar_resultados(
                prompt_usuario,
                resultados
            )

        st.markdown(
            f"""
            <div class="agent-box">
            {interpretacion}
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# NOTA DE SEGURIDAD Y ALCANCE
# ============================================================

st.markdown(
    """
    <div class="warning-box">
    <b>Alcance de la aplicación:</b><br><br>
    Los materiales encontrados son candidatos computacionales. La síntesis,
    estabilidad, toxicidad, biocompatibilidad y desempeño biomédico deben
    verificarse experimentalmente mediante protocolos validados.
    </div>
    """,
    unsafe_allow_html=True
)
