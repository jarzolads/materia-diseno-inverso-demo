import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Matéria: Agente para diseño inverso",
    page_icon="🔬",
    layout="wide"
)

st.title("Matéria: Agente para el diseño inverso de materiales")

st.markdown("### Desarrollado por Dr. Jesús Andrés Arzola Flores")

st.caption(
    "Aplicación demostrativa con LLM + Materials Project"
)

st.write(
    """
    Esta aplicación muestra cómo un agente de inteligencia artificial puede
    interpretar propiedades objetivo, consultar materiales candidatos y proponer
    una estrategia preliminar de síntesis.
    """
)

# ============================================================
# CONCEPTOS CIENTÍFICOS
# ============================================================

st.header("1. Conceptos científicos")

st.info(
    """
    Antes de elegir los valores de búsqueda, revisa el significado de cada
    propiedad. Estos parámetros permiten traducir una necesidad científica
    en criterios de selección de materiales.
    """
)

conceptos = {
    "Magnetización": """
    La magnetización describe el momento magnético por unidad de volumen de un
    material. En términos sencillos, indica qué tan intensamente responde un
    material frente a un campo magnético externo.

    En aplicaciones biomédicas, una magnetización elevada puede facilitar la
    manipulación de nanopartículas mediante campos magnéticos. Sin embargo, una
    magnetización alta por sí sola no garantiza que el material sea adecuado:
    también deben analizarse el tamaño de partícula, la estabilidad, la
    biocompatibilidad y la respuesta térmica.
    """,

    "Energía sobre el envolvente": """
    La energía sobre el envolvente, conocida como energy above hull o E_hull,
    indica la distancia energética de un material respecto a la envolvente de
    estabilidad termodinámica.

    Un valor cercano a cero sugiere que el material es relativamente estable
    frente a posibles descomposiciones hacia otras fases. Un valor mayor indica
    que puede ser menos estable. Esta propiedad no es una garantía absoluta de
    síntesis, pero sirve como indicador computacional inicial.
    """,

    "Densidad": """
    La densidad es la masa por unidad de volumen. Puede ser importante para
    aplicaciones donde se requiere dispersar, transportar o concentrar un
    material.

    En nanopartículas biomédicas, una densidad elevada puede influir en la
    sedimentación, la separación magnética y el comportamiento de las
    suspensiones.
    """,

    "Ordenamiento magnético": """
    El ordenamiento magnético describe la organización de los momentos
    magnéticos dentro del sólido.

    Algunos ejemplos son ferromagnético, antiferromagnético, ferrimagnético y
    paramagnético. Para aplicaciones de hipertermia magnética, normalmente se
    buscan materiales que respondan de forma controlada a un campo alterno.
    """,

    "Diseño inverso": """
    En el diseño directo se parte de una composición y se calculan sus
    propiedades.

    En el diseño inverso se parte de las propiedades deseadas y se buscan
    composiciones que potencialmente puedan cumplirlas.

    La aplicación sigue este flujo:

    1. El usuario define propiedades objetivo.
    2. Gemini interpreta la solicitud.
    3. Se consultan materiales candidatos.
    4. Se filtran los resultados.
    5. Se propone una estrategia experimental preliminar.
    """
}

for nombre, explicacion in conceptos.items():
    with st.expander(nombre, expanded=False):
        st.markdown(explicacion)

st.warning(
    """
    Materials Project proporciona propiedades calculadas para estructuras
    cristalinas. Estos datos no sustituyen la caracterización experimental ni
    garantizan que una nanopartícula sintetizada tenga exactamente las mismas
    propiedades.
    """
)

# ============================================================
# CONFIGURACIÓN DE APIS
# ============================================================

MP_API_KEY = st.secrets.get("MP_API_KEY", os.getenv("MP_API_KEY", ""))
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
)

# ============================================================
# DATOS DE DEMOSTRACIÓN
# ============================================================

DATOS_DEMO = pd.DataFrame([
    {
        "material_id": "mp-12767",
        "formula_pretty": "Fe3O4",
        "chemsys": "Fe-O",
        "density": 5.18,
        "energy_above_hull": 0.002,
        "ordering": "FiM",
        "total_magnetization_normalized_vol": 480.0,
        "symmetry": "Fd-3m"
    },
    {
        "material_id": "mp-19306",
        "formula_pretty": "Fe2O3",
        "chemsys": "Fe-O",
        "density": 5.24,
        "energy_above_hull": 0.015,
        "ordering": "AFM",
        "total_magnetization_normalized_vol": 25.0,
        "symmetry": "R-3c"
    },
    {
        "material_id": "mp-23115",
        "formula_pretty": "CoFe2O4",
        "chemsys": "Co-Fe-O",
        "density": 5.30,
        "energy_above_hull": 0.008,
        "ordering": "FiM",
        "total_magnetization_normalized_vol": 520.0,
        "symmetry": "Fd-3m"
    },
    {
        "material_id": "mp-1234",
        "formula_pretty": "NiFe2O4",
        "chemsys": "Fe-Ni-O",
        "density": 5.35,
        "energy_above_hull": 0.012,
        "ordering": "FiM",
        "total_magnetization_normalized_vol": 390.0,
        "symmetry": "Fd-3m"
    },
    {
        "material_id": "mp-5678",
        "formula_pretty": "MnFe2O4",
        "chemsys": "Fe-Mn-O",
        "density": 4.86,
        "energy_above_hull": 0.021,
        "ordering": "FiM",
        "total_magnetization_normalized_vol": 430.0,
        "symmetry": "Fd-3m"
    }
])

# ============================================================
# FUNCIONES
# ============================================================

def cargar_materials_project():
    """
    Consulta Materials Project cuando existe una clave válida.
    Si no existe, utiliza datos demostrativos.
    """

    if not MP_API_KEY:
        return DATOS_DEMO.copy(), "Se están utilizando datos demostrativos."

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
            documentos = mpr.materials.summary.search(
                chemsys="Fe-O",
                fields=campos,
                num_chunks=1
            )

        registros = []

        for doc in documentos:
            registros.append({
                "material_id": str(doc.material_id),
                "formula_pretty": doc.formula_pretty,
                "chemsys": doc.chemsys,
                "density": doc.density,
                "energy_above_hull": doc.energy_above_hull,
                "ordering": str(doc.ordering),
                "total_magnetization_normalized_vol":
                    doc.total_magnetization_normalized_vol,
                "symmetry": str(doc.symmetry)
            })

        datos = pd.DataFrame(registros)

        if datos.empty:
            return DATOS_DEMO.copy(), "La consulta no devolvió datos."

        return datos, "Datos cargados desde Materials Project."

    except Exception as error:
        return DATOS_DEMO.copy(), (
            "No fue posible consultar Materials Project. "
            f"Se utilizaron datos demostrativos. Detalle: {error}"
        )


def filtrar_materiales(
    datos,
    magnetizacion_minima,
    energia_maxima,
    densidad_maxima,
    ordenar_por
):
    resultado = datos.copy()

    resultado = resultado[
        resultado["total_magnetizacion_normalized_vol"].fillna(0)
        >= magnetizacion_minima
    ]

    resultado = resultado[
        resultado["energy_above_hull"].fillna(999)
        <= energia_maxima
    ]

    if densidad_maxima > 0:
        resultado = resultado[
            resultado["density"].fillna(999)
            <= densidad_maxima
        ]

    if ordenar_por == "Mayor magnetización":
        resultado = resultado.sort_values(
            "total_magnetization_normalized_vol",
            ascending=False
        )
    else:
        resultado = resultado.sort_values(
            "energy_above_hull",
            ascending=True
        )

    return resultado


def consultar_gemini(consulta, candidatos):
    """
    Gemini explica la solicitud y genera una recomendación textual.
    """

    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. La búsqueda se realizó mediante "
            "los filtros numéricos de la aplicación."
        )

    try:
        from google import genai

        cliente = genai.Client(api_key=GEMINI_API_KEY)

        datos = candidatos.head(10).to_dict(orient="records")

        prompt = f"""
        Actúa como un agente educativo de diseño inverso de materiales.

        Solicitud del usuario:
        {consulta}

        Materiales candidatos encontrados:
        {json.dumps(datos, ensure_ascii=False, default=str)}

        Explica en español:

        1. Qué propiedad parece ser prioritaria.
        2. Qué material o materiales son los candidatos más interesantes.
        3. Qué compromisos existen entre magnetización, estabilidad y densidad.
        4. Qué información experimental todavía debe verificarse.
        5. Una estrategia preliminar de síntesis para magnetita o ferritas,
           sin presentarla como un protocolo experimental validado.
        6. Qué caracterizaciones deberían realizarse.

        Sé claro, prudente y evita afirmar que el material es automáticamente
        biocompatible o adecuado para uso clínico.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible consultar Gemini: {error}"


# ============================================================
# SELECCIÓN DE PROPIEDADES
# ============================================================

st.header("2. Define las propiedades objetivo")

st.write(
    """
    Elige los valores que deseas utilizar para buscar candidatos. Estos valores
    representan una primera aproximación al problema de diseño inverso.
    """
)

columna_1, columna_2, columna_3 = st.columns(3)

with columna_1:
    magnetizacion_minima = st.number_input(
        "Magnetización mínima",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=10.0,
        help="Valor mínimo de magnetización normalizada por volumen."
    )

with columna_2:
    energia_maxima = st.number_input(
        "Energía sobre el envolvente máxima",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.005,
        format="%.3f",
        help="Valores cercanos a cero indican mayor estabilidad energética."
    )

with columna_3:
    densidad_maxima = st.number_input(
        "Densidad máxima",
        min_value=0.0,
        max_value=15.0,
        value=6.0,
        step=0.1,
        help="Usa 0 si no deseas aplicar un filtro de densidad."
    )

ordenar_por = st.selectbox(
    "Ordenar resultados por",
    [
        "Mayor magnetización",
        "Menor energía sobre el envolvente"
    ]
)

consulta_usuario = st.text_area(
    "Describe el objetivo del material",
    value=(
        "Busco un material magnético estable, con magnetización elevada, "
        "para explorar una aplicación de hipertermia magnética."
    ),
    height=100
)

# ============================================================
# BOTONES DE EJECUCIÓN
# ============================================================

columna_a, columna_b = st.columns(2)

with columna_a:
    cargar = st.button(
        "Cargar datos de Materials Project",
        use_container_width=True
    )

with columna_b:
    ejemplo = st.button(
        "Probar ejemplo de diseño inverso",
        use_container_width=True
    )

if "datos" not in st.session_state:
    st.session_state.datos = DATOS_DEMO.copy()

if cargar or ejemplo:
    with st.spinner("Consultando materiales..."):
        datos, mensaje = cargar_materials_project()
        st.session_state.datos = datos

    st.success(mensaje)

datos = st.session_state.datos

# ============================================================
# RESULTADOS DEL DISEÑO INVERSO
# ============================================================

if cargar or ejemplo:
    resultados = filtrar_materiales(
        datos,
        magnetizacion_minima,
        energia_maxima,
        densidad_maxima,
        ordenar_por
    )

    st.header("3. Resultados del diseño inverso")

    if resultados.empty:
        st.error(
            "No se encontraron materiales con los criterios seleccionados. "
            "Prueba aumentando la energía máxima o reduciendo la magnetización mínima."
        )
    else:
        st.success(
            f"Se encontraron {len(resultados)} materiales candidatos."
        )

        columnas_mostrar = [
            "material_id",
            "formula_pretty",
            "chemsys",
            "density",
            "energy_above_hull",
            "ordering",
            "total_magnetization_normalized_vol",
            "symmetry"
        ]

        st.dataframe(
            resultados[columnas_mostrar],
            use_container_width=True,
            hide_index=True
        )

        grafica = px.scatter(
            resultados,
            x="energy_above_hull",
            y="total_magnetization_normalized_vol",
            size="density",
            color="ordering",
            hover_name="formula_pretty",
            hover_data=["material_id", "chemsys"],
            labels={
                "energy_above_hull": "Energía sobre el envolvente",
                "total_magnetization_normalized_vol":
                    "Magnetización normalizada por volumen",
                "ordering": "Ordenamiento magnético",
                "density": "Densidad"
            },
            title="Mapa de candidatos para diseño inverso"
        )

        st.plotly_chart(
            grafica,
            use_container_width=True
        )

        st.header("4. Interpretación del agente")

        with st.spinner("Gemini está interpretando los resultados..."):
            explicacion = consultar_gemini(
                consulta_usuario,
                resultados
            )

        st.markdown(explicacion)

# ============================================================
# ESTRATEGIA DE SÍNTESIS
# ============================================================

st.header("5. Estrategia preliminar de síntesis")

st.write(
    """
    Para la demostración se utiliza como ejemplo la síntesis de magnetita
    (Fe3O4) mediante coprecipitación de sales de hierro. Esta sección sirve
    para mostrar cómo un agente puede convertir un candidato computacional en
    una propuesta experimental organizada.
    """
)

st.subheader("Reactivos químicos")

reactivos = pd.DataFrame([
    {
        "Reactivo": "FeCl3·6H2O",
        "Función": "Fuente de Fe(III)",
        "Observación": "Revisar hoja de seguridad"
    },
    {
        "Reactivo": "FeCl2·4H2O",
        "Función": "Fuente de Fe(II)",
        "Observación": "Proteger de oxidación excesiva"
    },
    {
        "Reactivo": "NH4OH o NaOH",
        "Función": "Agente precipitante",
        "Observación": "Agregar de forma gradual"
    },
    {
        "Reactivo": "Agua desionizada",
        "Función": "Disolución y lavado",
        "Observación": "Usar agua limpia"
    },
    {
        "Reactivo": "Etanol",
        "Función": "Lavado opcional",
        "Observación": "Inflamable"
    }
])

st.dataframe(
    reactivos,
    use_container_width=True,
    hide_index=True
)

st.subheader("Materiales y equipo de laboratorio")

equipo = pd.DataFrame([
    {
        "Equipo": "Balanza analítica",
        "Uso": "Pesaje de los precursores"
    },
    {
        "Equipo": "Matraces aforados",
        "Uso": "Preparación de soluciones"
    },
    {
        "Equipo": "Vasos de precipitados",
        "Uso": "Mezcla de reactivos"
    },
    {
        "Equipo": "Parrilla con agitación magnética",
        "Uso": "Agitación y control de temperatura"
    },
    {
        "Equipo": "Barras magnéticas de PTFE",
        "Uso": "Agitación de las soluciones"
    },
    {
        "Equipo": "pH-metro o tiras de pH",
        "Uso": "Control del pH"
    },
    {
        "Equipo": "Jeringa, bureta o embudo de adición",
        "Uso": "Adición controlada de la base"
    },
    {
        "Equipo": "Termómetro",
        "Uso": "Monitoreo de la temperatura"
    },
    {
        "Equipo": "Centrífuga o filtración al vacío",
        "Uso": "Separación del sólido"
    },
    {
        "Equipo": "Papel filtro y embudo Büchner",
        "Uso": "Recuperación de nanopartículas"
    },
    {
        "Equipo": "Estufa u horno de vacío",
        "Uso": "Secado del material"
    },
    {
        "Equipo": "Desecador",
        "Uso": "Almacenamiento de la muestra"
    },
    {
        "Equipo": "Campana de extracción",
        "Uso": "Manipulación segura de reactivos"
    },
    {
        "Equipo": "Bata, guantes y gafas",
        "Uso": "Equipo de protección personal"
    },
    {
        "Equipo": "Recipientes para residuos",
        "Uso": "Separación de residuos químicos"
    }
])

st.dataframe(
    equipo,
    use_container_width=True,
    hide_index=True
)

st.subheader("Procedimiento paso a paso")

pasos_sintesis = [
    "Revisar las hojas de seguridad y preparar el área de trabajo con el equipo de protección personal correspondiente.",

    "Definir la escala de síntesis y calcular la cantidad de FeCl3·6H2O y FeCl2·4H2O de acuerdo con el protocolo experimental seleccionado.",

    "Pesar las sales de hierro utilizando una balanza analítica.",

    "Disolver por separado los precursores en agua desionizada.",

    "Preparar la solución de NH4OH o NaOH en un recipiente independiente.",

    "Colocar la solución de hierro bajo agitación magnética y controlar la temperatura.",

    "Agregar lentamente el agente precipitante mientras se monitorea el pH.",

    "Mantener la agitación durante el tiempo de envejecimiento especificado por el protocolo.",

    "Separar las nanopartículas mediante centrifugación o filtración al vacío.",

    "Lavar el sólido varias veces con agua desionizada para eliminar especies solubles.",

    "Realizar un lavado opcional con etanol si el protocolo seleccionado lo indica.",

    "Secar el sólido en una estufa o bajo vacío utilizando condiciones validadas.",

    "Guardar la muestra en un desecador y registrar fecha, composición y condiciones de síntesis.",

    "Caracterizar el material mediante difracción de rayos X, microscopía, distribución de tamaño y magnetometría.",

    "Comparar las propiedades experimentales con los objetivos planteados por el agente."
]

for numero, paso in enumerate(pasos_sintesis, start=1):
    st.markdown(f"**Paso {numero}.** {paso}")

st.warning(
    """
    La síntesis mostrada es una estrategia educativa preliminar. Las
    concentraciones, cantidades, temperaturas, tiempos y condiciones exactas
    deben tomarse de un protocolo validado en la literatura y aprobado por el
    laboratorio. La aplicación no determina por sí sola biocompatibilidad,
    seguridad clínica ni desempeño terapéutico.
    """
)

# ============================================================
# COSTO PRELIMINAR
# ============================================================

st.header("6. Costo preliminar")

costos = pd.DataFrame([
    {
        "Material": "FeCl3·6H2O",
        "Cantidad_g": 10,
        "Precio_paquete_MXN": 450,
        "Paquete_g": 100
    },
    {
        "Material": "FeCl2·4H2O",
        "Cantidad_g": 10,
        "Precio_paquete_MXN": 650,
        "Paquete_g": 100
    },
    {
        "Material": "NH4OH",
        "Cantidad_g": 50,
        "Precio_paquete_MXN": 300,
        "Paquete_g": 500
    },
    {
        "Material": "Agua desionizada",
        "Cantidad_g": 500,
        "Precio_paquete_MXN": 80,
        "Paquete_g": 5000
    },
    {
        "Material": "Etanol",
        "Cantidad_g": 100,
        "Precio_paquete_MXN": 180,
        "Paquete_g": 1000
    },
    {
        "Material": "Filtros y consumibles",
        "Cantidad_g": 1,
        "Precio_paquete_MXN": 250,
        "Paquete_g": 1
    }
])

costos["Costo_estimado_MXN"] = (
    costos["Cantidad_g"]
    / costos["Paquete_g"]
    * costos["Precio_paquete_MXN"]
)

st.dataframe(
    costos,
    use_container_width=True,
    hide_index=True
)

costo_total = costos["Costo_estimado_MXN"].sum()

st.metric(
    "Costo preliminar de reactivos y consumibles",
    f"${costo_total:,.2f} MXN"
)

st.caption(
    "Los precios son ilustrativos y deben sustituirse por cotizaciones reales."
)
