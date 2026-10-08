import os
import json
import re
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
    layout="wide"
)

# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        color: #0B3D91;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #555555;
        font-size: 20px;
        margin-bottom: 20px;
    }

    .section-title {
        color: #0B3D91;
        border-bottom: 2px solid #4EA5D9;
        padding-bottom: 6px;
        margin-top: 28px;
    }

    .metric-card {
        background-color: #F2F7FB;
        border-left: 5px solid #0B3D91;
        padding: 16px;
        border-radius: 10px;
        min-height: 120px;
        text-align: center;
    }

    .metric-card h4 {
        color: #456;
        margin-bottom: 8px;
    }

    .metric-card h2 {
        color: #0B3D91;
    }

    .agent-card {
        background-color: #F8FAFC;
        border: 1px solid #D8E2EC;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }

    .warning-card {
        background-color: #FFF8E7;
        border-left: 5px solid #E7A900;
        padding: 15px;
        border-radius: 8px;
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

st.info(
    """
    **¿Para qué sirve Matéria?**

    Matéria es un agente de inteligencia artificial que permite buscar
    materiales a partir de las propiedades que se desean obtener.

    El usuario puede escribir un objetivo científico mediante un prompt o
    seleccionar valores numéricos. El agente interpreta la solicitud, consulta
    Materials Project, filtra los candidatos y genera una interpretación
    científica preliminar.
    """
)

st.caption(
    "La aplicación consulta datos científicos; no utiliza una lista fija de materiales."
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
# CONCEPTOS
# ============================================================

st.markdown(
    '<div class="section-title">Conceptos científicos</div>',
    unsafe_allow_html=True
)

with st.expander("Magnetización"):
    st.write(
        """
        La magnetización es el momento magnético por unidad de volumen.
        Describe la intensidad con la que un material responde frente a un campo
        magnético externo.

        En hipertermia magnética puede relacionarse con la respuesta ante un
        campo alterno, aunque también deben evaluarse el tamaño de partícula,
        la estabilidad coloidal, la toxicidad y la biocompatibilidad.
        """
    )

with st.expander("Energía sobre el envolvente"):
    st.write(
        """
        La energía sobre el envolvente, o `energy above hull`, es un indicador
        computacional de estabilidad termodinámica.

        Un valor cercano a cero indica que el material se encuentra cerca de la
        envolvente de estabilidad. No garantiza por sí solo que el material
        pueda sintetizarse.
        """
    )

with st.expander("Densidad"):
    st.write(
        """
        La densidad es la masa por unidad de volumen. Puede influir en la
        sedimentación, separación magnética y preparación de suspensiones.
        """
    )

with st.expander("Diseño inverso"):
    st.write(
        """
        En el diseño directo se parte de una composición y se calculan sus
        propiedades.

        En el diseño inverso se parte de las propiedades deseadas y se buscan
        composiciones que puedan cumplirlas.

        La aplicación sigue este flujo:

        1. El usuario define un objetivo.
        2. Gemini interpreta el prompt.
        3. Materials Project proporciona candidatos.
        4. La aplicación filtra los resultados.
        5. Gemini interpreta los candidatos.
        6. Se propone una estrategia experimental preliminar.
        """
    )

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limpiar_json(texto):
    """
    Extrae un objeto JSON aunque Gemini lo devuelva dentro de markdown.
    """

    texto = texto.strip()
    texto = texto.replace("```json", "")
    texto = texto.replace("```", "")
    texto = texto.strip()

    inicio = texto.find("{")
    final = texto.rfind("}")

    if inicio == -1 or final == -1:
        raise ValueError("Gemini no devolvió un JSON válido.")

    return json.loads(texto[inicio:final + 1])


def interpretar_prompt(prompt):
    """
    Utiliza Gemini para convertir un prompt científico en filtros numéricos.
    """

    filtros_defecto = {
        "magnetizacion_minima": 0.0,
        "energia_maxima": 0.10,
        "densidad_maxima": 0.0,
        "elementos": [],
        "sistema_quimico": "",
        "ordenar_por": "energy_above_hull",
        "objetivo": prompt
    }

    if not GEMINI_API_KEY:
        return filtros_defecto, (
            "Gemini no está conectado. Se utilizaron los valores manuales."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        instrucciones = f"""
        Analiza el siguiente objetivo de diseño inverso de materiales:

        {prompt}

        Devuelve exclusivamente un objeto JSON válido con esta estructura:

        {{
            "magnetizacion_minima": número,
            "energia_maxima": número,
            "densidad_maxima": número,
            "elementos": ["Fe", "O"],
            "sistema_quimico": "",
            "ordenar_por": "magnetizacion" o "energy_above_hull",
            "objetivo": "resumen breve"
        }}

        Reglas:

        - Si no se especifica magnetización, usa 0.
        - Si no se especifica energía, usa 0.10.
        - Si no se especifica densidad, usa 0.
        - Si se mencionan elementos, escríbelos con símbolos químicos.
        - No inventes propiedades experimentales.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instrucciones
        )

        datos = limpiar_json(respuesta.text)

        for clave, valor in filtros_defecto.items():
            if clave not in datos:
                datos[clave] = valor

        return datos, "Gemini interpretó el prompt."

    except Exception as error:
        return filtros_defecto, (
            f"No fue posible interpretar el prompt con Gemini: {error}"
        )


def preparar_datos(datos):
    """
    Garantiza que las columnas necesarias existan.
    """

    datos = datos.copy()

    columnas = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "density",
        "energy_above_hull",
        "ordering",
        "total_magnetization_normalized_vol",
        "symmetry"
    ]

    for columna in columnas:
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


def consultar_materials_project(
    elementos,
    sistema_quimico,
    limite_resultados
):
    """
    Consulta Materials Project sin utilizar materiales predefinidos.
    """

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

        with MPRester(MP_API_KEY) as mpr:

            argumentos = {
                "fields": campos,
                "num_chunks": 1
            }

            if sistema_quimico.strip():
                argumentos["chemsys"] = sistema_quimico.strip()

            elif elementos:
                argumentos["elements"] = elementos

            documentos = mpr.materials.summary.search(
                **argumentos
            )

        registros = []

        for doc in documentos[:limite_resultados]:
            registros.append({
                "material_id": str(
                    getattr(doc, "material_id", "")
                ),
                "formula_pretty": getattr(
                    doc, "formula_pretty", ""
                ),
                "chemsys": getattr(
                    doc, "chemsys", ""
                ),
                "density": getattr(
                    doc, "density", np.nan
                ),
                "energy_above_hull": getattr(
                    doc, "energy_above_hull", np.nan
                ),
                "ordering": str(
                    getattr(doc, "ordering", "No disponible")
                ),
                "total_magnetization_normalized_vol": getattr(
                    doc,
                    "total_magnetization_normalized_vol",
                    np.nan
                ),
                "symmetry": str(
                    getattr(doc, "symmetry", "No disponible")
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
    """
    Gemini interpreta exclusivamente los materiales devueltos por Materials
    Project.
    """

    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. La tabla muestra directamente los "
            "resultados filtrados de Materials Project."
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
        Interpreta estos resultados de Materials Project para el objetivo:

        {prompt}

        Resultados:
        {json.dumps(registros, ensure_ascii=False, default=str)}

        Responde en español con estas secciones:

        1. Interpretación del objetivo.
        2. Candidatos más interesantes.
        3. Compromisos entre estabilidad y propiedades.
        4. Limitaciones de los datos calculados.
        5. Caracterización experimental recomendada.
        6. Estrategia preliminar de síntesis.

        No afirmes que los materiales son biocompatibles ni clínicamente seguros.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=solicitud
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible interpretar los resultados: {error}"


# ============================================================
# PROMPT DEL USUARIO
# ============================================================

st.markdown(
    '<div class="section-title">Define tu objetivo de diseño</div>',
    unsafe_allow_html=True
)

prompt_usuario = st.text_area(
    "Escribe una solicitud científica",
    value=(
        "Busca materiales magnéticos estables basados en hierro, con alta "
        "magnetización, para explorar una aplicación de hipertermia."
    ),
    height=130,
    help=(
        "Puedes escribir la aplicación, los elementos o las propiedades "
        "deseadas."
    )
)

st.markdown("### Parámetros de búsqueda")

columna_1, columna_2, columna_3 = st.columns(3)

with columna_1:
    magnetizacion_manual = st.number_input(
        "Magnetización mínima",
        min_value=0.0,
        max_value=2000.0,
        value=300.0,
        step=10.0
    )

with columna_2:
    energia_manual = st.number_input(
        "Energía máxima sobre el envolvente",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.01,
        format="%.3f"
    )

with columna_3:
    densidad_manual = st.number_input(
        "Densidad máxima",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        help="Usa 0 para no aplicar filtro de densidad."
    )

st.markdown("### Opciones de consulta")

columna_4, columna_5 = st.columns(2)

with columna_4:
    elementos_texto = st.text_input(
        "Elementos opcionales",
        value="Fe, O",
        help="Ejemplo: Fe, O o Li, Co, O"
    )

with columna_5:
    sistema_quimico = st.text_input(
        "Sistema químico opcional",
        value="",
        help="Ejemplo: Fe-O. Déjalo vacío para usar elementos."
    )

limite_resultados = st.slider(
    "Número máximo de resultados consultados",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

usar_prompt = st.checkbox(
    "Permitir que Gemini interprete automáticamente el prompt",
    value=True
)

# ============================================================
# EJECUCIÓN
# ============================================================

buscar = st.button(
    "🔎 Buscar materiales realmente en Materials Project",
    type="primary",
    use_container_width=True
)

if buscar:

    if not MP_API_KEY:
        st.error(
            """
            No se encontró `MP_API_KEY`.

            En Streamlit Cloud ve a:

            `Manage app → Settings → Secrets`

            y agrega:

            `MP_API_KEY = "tu_clave_de_materials_project"`
            """
        )
        st.stop()

    if usar_prompt:
        filtros, mensaje_gemini = interpretar_prompt(
            prompt_usuario
        )
        st.info(mensaje_gemini)

        magnetizacion_busqueda = float(
            filtros.get(
                "magnetizacion_minima",
                magnetizacion_manual
            )
        )

        energia_busqueda = float(
            filtros.get(
                "energia_maxima",
                energia_manual
            )
        )

        densidad_busqueda = float(
            filtros.get(
                "densidad_maxima",
                densidad_manual
            )
        )

        elementos_prompt = filtros.get(
            "elementos",
            []
        )

        sistema_prompt = filtros.get(
            "sistema_quimico",
            ""
        )

        if elementos_prompt:
            elementos = elementos_prompt
        else:
            elementos = [
                elemento.strip()
                for elemento in elementos_texto.split(",")
                if elemento.strip()
            ]

        if sistema_prompt:
            sistema_consulta = sistema_prompt
        else:
            sistema_consulta = sistema_quimico

    else:
        magnetizacion_busqueda = magnetizacion_manual
        energia_busqueda = energia_manual
        densidad_busqueda = densidad_manual

        elementos = [
            elemento.strip()
            for elemento in elementos_texto.split(",")
            if elemento.strip()
        ]

        sistema_consulta = sistema_quimico

    with st.spinner(
        "Consultando directamente Materials Project..."
    ):
        datos, mensaje = consultar_materials_project(
            elementos=elementos,
            sistema_quimico=sistema_consulta,
            limite_resultados=limite_resultados
        )

    st.success(mensaje)

    if datos.empty:
        st.error(
            """
            No se obtuvieron registros.

            Revisa que:

            - `MP_API_KEY` esté correctamente configurada.
            - El sistema químico exista.
            - Los elementos estén escritos con símbolos químicos.
            - La consulta no sea demasiado restrictiva.
            """
        )
        st.stop()

    resultados = filtrar_materiales(
        datos,
        magnetizacion_busqueda,
        energia_busqueda,
        densidad_busqueda
    )

    st.markdown(
        '<div class="section-title">Resultados del diseño inverso</div>',
        unsafe_allow_html=True
    )

    if resultados.empty:
        st.warning(
            """
            Materials Project sí devolvió materiales, pero ninguno cumple
            simultáneamente los valores seleccionados.

            Prueba aumentando la energía máxima, reduciendo la magnetización
            mínima o eliminando el filtro de densidad.
            """
        )

        st.write(
            "Número de registros antes del filtrado:",
            len(datos)
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
                    <h4>Menor E_hull</h4>
                    <h2>{resultados["energy_above_hull"].min():.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tarjeta_3:
            magnetizaciones = resultados[
                "total_magnetization_normalized_vol"
            ].dropna()

            if len(magnetizaciones) > 0:
                valor_magnetizacion = f"{magnetizaciones.max():.2f}"
            else:
                valor_magnetizacion = "N/D"

            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mayor magnetización</h4>
                    <h2>{valor_magnetizacion}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tarjeta_4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Sistemas químicos</h4>
                    <h2>{resultados["chemsys"].nunique()}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.dataframe(
            resultados,
            use_container_width=True,
            hide_index=True
        )

        columnas_grafica = [
            "energy_above_hull",
            "total_magnetization_normalized_vol",
            "formula_pretty",
            "density",
            "chemsys"
        ]

        grafica_datos = resultados[
            columnas_grafica
        ].dropna(
            subset=[
                "energy_above_hull",
                "total_magnetization_normalized_vol"
            ]
        )

        if not grafica_datos.empty:
            grafica = px.scatter(
                grafica_datos,
                x="energy_above_hull",
                y="total_magnetization_normalized_vol",
                size="density",
                color="chemsys",
                hover_name="formula_pretty",
                labels={
                    "energy_above_hull":
                        "Energía sobre el envolvente",
                    "total_magnetization_normalized_vol":
                        "Magnetización normalizada por volumen",
                    "density":
                        "Densidad",
                    "chemsys":
                        "Sistema químico"
                },
                title="Mapa de candidatos encontrados"
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
            '<div class="section-title">Interpretación de Gemini</div>',
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
            <div class="agent-card">
            {interpretacion}
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# NOTA FINAL
# ============================================================

st.markdown(
    """
    <div class="warning-card">
    <b>Importante:</b> los candidatos obtenidos son materiales computacionales
    identificados mediante criterios de búsqueda. La síntesis, la estabilidad
    de nanopartículas, la toxicidad y el desempeño biomédico deben validarse
    experimentalmente.
    </div>
    """,
    unsafe_allow_html=True
)
