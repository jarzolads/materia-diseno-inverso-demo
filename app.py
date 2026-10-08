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
    page_title="Matéria: Agente para diseño inverso",
    page_icon="🔬",
    layout="wide"
)

st.title("Matéria: Agente para el diseño inverso de materiales")
st.markdown("### Desarrollado por Jesús Arzola")

st.info(
    """
    **¿Para qué sirve Matéria?**

    Matéria es un agente educativo para las primeras etapas del diseño inverso
    de materiales. El usuario define las propiedades que desea y la aplicación
    busca materiales candidatos, compara sus características y propone una
    estrategia preliminar de síntesis.

    El agente puede ayudar a responder preguntas como:

    - ¿Qué materiales tienen una magnetización elevada?
    - ¿Qué candidatos son relativamente estables?
    - ¿Qué composición podría ser interesante para una aplicación?
    - ¿Qué reactivos y equipo se necesitarían para explorar su síntesis?
    - ¿Qué caracterizaciones deberían realizarse?

    Matéria no sustituye la validación experimental, la revisión bibliográfica,
    la caracterización ni la evaluación de seguridad.
    """
)

st.caption("Demostración educativa con Gemini y Materials Project")

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
# TIPOS DE MATERIALES
# ============================================================

st.header("1. ¿Para qué tipos de materiales funciona?")

st.markdown(
    """
    Matéria funciona principalmente con materiales sólidos inorgánicos que
    pueden representarse mediante una composición química y una estructura
    cristalina.

    Es especialmente útil para explorar:
    """
)

materiales = pd.DataFrame([
    {
        "Tipo de material": "Óxidos metálicos",
        "Ejemplos": "Fe3O4, Fe2O3, TiO2, ZnO",
        "Aplicaciones": "Magnetismo, catálisis, sensores y energía"
    },
    {
        "Tipo de material": "Ferritas",
        "Ejemplos": "CoFe2O4, NiFe2O4, MnFe2O4",
        "Aplicaciones": "Magnetismo, hipertermia y separación magnética"
    },
    {
        "Tipo de material": "Aleaciones metálicas",
        "Ejemplos": "Fe-Ni, Fe-Co, Cu-Zn",
        "Aplicaciones": "Propiedades mecánicas, magnéticas y térmicas"
    },
    {
        "Tipo de material": "Materiales cerámicos",
        "Ejemplos": "Al2O3, ZrO2, BaTiO3",
        "Aplicaciones": "Aislamiento, sensores y materiales dieléctricos"
    },
    {
        "Tipo de material": "Materiales para energía",
        "Ejemplos": "Materiales para baterías y electrodos",
        "Aplicaciones": "Almacenamiento y conversión de energía"
    },
    {
        "Tipo de material": "Materiales bidimensionales",
        "Ejemplos": "Grafeno, MoS2 y otros dicalcogenuros",
        "Aplicaciones": "Electrónica, sensores y catálisis"
    }
])

st.dataframe(
    materiales,
    use_container_width=True,
    hide_index=True
)

st.warning(
    """
    Esta demostración está configurada para materiales magnéticos basados en
    hierro y óxidos metálicos. Para utilizar polímeros, proteínas, moléculas
    orgánicas o materiales compuestos sería necesario modificar las propiedades,
    la base de datos y la estrategia de síntesis.
    """
)

# ============================================================
# CONCEPTOS CIENTÍFICOS
# ============================================================

st.header("2. Conceptos científicos")

with st.expander("Magnetización"):
    st.markdown(
        """
        La magnetización es el momento magnético por unidad de volumen de un
        material. Indica qué tan intensamente responde frente a un campo
        magnético externo.

        Para hipertermia magnética puede ser una propiedad importante porque
        influye en la respuesta ante un campo magnético alterno. Sin embargo,
        también deben estudiarse el tamaño de partícula, la estabilidad, la
        dispersión, la toxicidad y la biocompatibilidad.
        """
    )

with st.expander("Energía sobre el envolvente"):
    st.markdown(
        """
        La energía sobre el envolvente, conocida como `energy above hull` o
        `E_hull`, es un indicador computacional de estabilidad termodinámica.

        Un valor cercano a cero indica que el material está cerca de la
        envolvente de estabilidad. Un valor mayor puede indicar que existen
        otras fases más estables.
        """
    )

with st.expander("Densidad"):
    st.markdown(
        """
        La densidad es la masa por unidad de volumen. Puede influir en la
        sedimentación, separación magnética y preparación de suspensiones.
        """
    )

with st.expander("Ordenamiento magnético"):
    st.markdown(
        """
        Describe cómo se organizan los momentos magnéticos dentro del sólido.

        Algunos tipos son ferromagnético, antiferromagnético, ferrimagnético y
        paramagnético.
        """
    )

with st.expander("Diseño inverso"):
    st.markdown(
        """
        En el diseño directo se parte de una composición y se calculan sus
        propiedades.

        En el diseño inverso se parte de las propiedades deseadas y se buscan
        composiciones que potencialmente puedan cumplirlas.

        El flujo de trabajo es:

        1. Definir propiedades objetivo.
        2. Consultar una base de datos.
        3. Filtrar materiales candidatos.
        4. Comparar los resultados.
        5. Proponer una síntesis preliminar.
        6. Validar experimentalmente.
        """
    )

st.warning(
    """
    Materials Project proporciona principalmente propiedades calculadas para
    estructuras cristalinas. Los resultados no garantizan que una nanopartícula
    sintetizada tenga exactamente las mismas propiedades.
    """
)

# ============================================================
# DATOS DE DEMOSTRACIÓN
# ============================================================

DATOS_DEMO = pd.DataFrame([
    {
        "material_id": "mp-demo-001",
        "formula_pretty": "Fe3O4",
        "chemsys": "Fe-O",
        "density": 5.18,
        "energy_above_hull": 0.002,
        "ordering": "Ferrimagnético",
        "total_magnetization_normalized_vol": 480.0,
        "symmetry": "Fd-3m"
    },
    {
        "material_id": "mp-demo-002",
        "formula_pretty": "Fe2O3",
        "chemsys": "Fe-O",
        "density": 5.24,
        "energy_above_hull": 0.015,
        "ordering": "Antiferromagnético",
        "total_magnetization_normalized_vol": 25.0,
        "symmetry": "R-3c"
    },
    {
        "material_id": "mp-demo-003",
        "formula_pretty": "CoFe2O4",
        "chemsys": "Co-Fe-O",
        "density": 5.30,
        "energy_above_hull": 0.008,
        "ordering": "Ferrimagnético",
        "total_magnetization_normalized_vol": 520.0,
        "symmetry": "Fd-3m"
    },
    {
        "material_id": "mp-demo-004",
        "formula_pretty": "NiFe2O4",
        "chemsys": "Fe-Ni-O",
        "density": 5.35,
        "energy_above_hull": 0.012,
        "ordering": "Ferrimagnético",
        "total_magnetization_normalized_vol": 390.0,
        "symmetry": "Fd-3m"
    },
    {
        "material_id": "mp-demo-005",
        "formula_pretty": "MnFe2O4",
        "chemsys": "Fe-Mn-O",
        "density": 4.86,
        "energy_above_hull": 0.021,
        "ordering": "Ferrimagnético",
        "total_magnetization_normalized_vol": 430.0,
        "symmetry": "Fd-3m"
    }
])

# ============================================================
# FUNCIONES
# ============================================================

def preparar_columnas(datos):
    resultado = datos.copy()

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
        if columna not in resultado.columns:
            resultado[columna] = np.nan

    columnas_numericas = [
        "density",
        "energy_above_hull",
        "total_magnetization_normalized_vol"
    ]

    for columna in columnas_numericas:
        resultado[columna] = pd.to_numeric(
            resultado[columna],
            errors="coerce"
        )

    return resultado


def cargar_materials_project():
    if not MP_API_KEY:
        return (
            DATOS_DEMO.copy(),
            "Se están utilizando datos demostrativos."
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
                "total_magnetization_normalized_vol": np.nan,
                "symmetry": str(
                    getattr(doc, "symmetry", "No disponible")
                )
            })

        datos = preparar_columnas(
            pd.DataFrame(registros)
        )

        if datos.empty:
            return (
                DATOS_DEMO.copy(),
                "No se encontraron datos. Se utilizaron datos demostrativos."
            )

        magnetizacion_disponible = (
            datos["total_magnetization_normalized_vol"]
            .notna()
            .any()
        )

        if not magnetizacion_disponible:
            return (
                DATOS_DEMO.copy(),
                """
                Materials Project no devolvió magnetización para esta consulta.
                Se utilizaron datos demostrativos para ejecutar el agente.
                """
            )

        return datos, "Datos cargados desde Materials Project."

    except Exception as error:
        return (
            DATOS_DEMO.copy(),
            "Se utilizaron datos demostrativos. "
            f"Detalle de la consulta: {error}"
        )


def filtrar_materiales(
    datos,
    magnetizacion_minima,
    energia_maxima,
    densidad_maxima,
    ordenar_por
):
    resultado = preparar_columnas(datos)

    magnetizacion = resultado[
        "total_magnetization_normalized_vol"
    ].fillna(0)

    energia = resultado[
        "energy_above_hull"
    ].fillna(999)

    densidad = resultado[
        "density"
    ].fillna(999)

    resultado = resultado[
        magnetizacion >= magnetizacion_minima
    ]

    resultado = resultado[
        energia.loc[resultado.index] <= energia_maxima
    ]

    if densidad_maxima > 0:
        resultado = resultado[
            densidad.loc[resultado.index] <= densidad_maxima
        ]

    if ordenar_por == "Mayor magnetización":
        resultado = resultado.sort_values(
            by="total_magnetization_normalized_vol",
            ascending=False
        )
    else:
        resultado = resultado.sort_values(
            by="energy_above_hull",
            ascending=True
        )

    return resultado


def consultar_gemini(consulta, candidatos):
    if not GEMINI_API_KEY:
        return (
            "Gemini no está conectado. La selección se realizó mediante "
            "los filtros numéricos."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        datos = candidatos.head(10).to_dict(
            orient="records"
        )

        prompt = f"""
        Actúa como un agente educativo de diseño inverso de materiales.

        Solicitud:
        {consulta}

        Candidatos:
        {json.dumps(datos, ensure_ascii=False, default=str)}

        Responde en español:

        1. Identifica la propiedad prioritaria.
        2. Explica cuál candidato es más interesante.
        3. Describe los compromisos entre magnetización, estabilidad y densidad.
        4. Indica qué información experimental falta.
        5. Propón una estrategia preliminar de síntesis.
        6. Indica qué técnicas de caracterización deben utilizarse.

        No afirmes que el material es automáticamente biocompatible o clínico.
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

st.header("3. Define las propiedades objetivo")

columna_1, columna_2, columna_3 = st.columns(3)

with columna_1:
    magnetizacion_minima = st.number_input(
        "Magnetización mínima",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=10.0
    )

with columna_2:
    energia_maxima = st.number_input(
        "Energía sobre el envolvente máxima",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.005,
        format="%.3f"
    )

with columna_3:
    densidad_maxima = st.number_input(
        "Densidad máxima",
        min_value=0.0,
        max_value=15.0,
        value=6.0,
        step=0.1
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
        "Busco un material magnético estable con magnetización elevada "
        "para explorar hipertermia magnética."
    ),
    height=100
)

# ============================================================
# BOTONES
# ============================================================

columna_a, columna_b = st.columns(2)

with columna_a:
    boton_cargar = st.button(
        "Cargar datos de Materials Project",
        use_container_width=True
    )

with columna_b:
    boton_ejemplo = st.button(
        "Probar ejemplo",
        use_container_width=True
    )

if "datos" not in st.session_state:
    st.session_state.datos = DATOS_DEMO.copy()

if boton_cargar or boton_ejemplo:
    with st.spinner("Consultando materiales..."):
        datos, mensaje = cargar_materials_project()
        st.session_state.datos = preparar_columnas(datos)

    st.success(mensaje)

# ============================================================
# RESULTADOS
# ============================================================

if boton_cargar or boton_ejemplo:

    resultados = filtrar_materiales(
        st.session_state.datos,
        magnetizacion_minima,
        energia_maxima,
        densidad_maxima,
        ordenar_por
    )

    st.header("4. Resultados del diseño inverso")

    if resultados.empty:
        st.error(
            """
            No se encontraron materiales con estos criterios.

            Prueba reduciendo la magnetización mínima o aumentando la energía
            sobre el envolvente máxima.
            """
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
            title="Mapa de candidatos"
        )

        st.plotly_chart(
            grafica,
            use_container_width=True
        )

        st.header("5. Interpretación del agente")

        with st.spinner("Gemini está interpretando los resultados..."):
            explicacion = consultar_gemini(
                consulta_usuario,
                resultados
            )

        st.markdown(explicacion)

# ============================================================
# SÍNTESIS
# ============================================================

st.header("6. Estrategia preliminar de síntesis")

st.write(
    """
    Como ejemplo se propone la síntesis de magnetita (Fe3O4) mediante
    coprecipitación de sales de hierro.
    """
)

st.subheader("Reactivos químicos")

reactivos = pd.DataFrame([
    {
        "Reactivo": "FeCl3·6H2O",
        "Función": "Fuente de Fe(III)"
    },
    {
        "Reactivo": "FeCl2·4H2O",
        "Función": "Fuente de Fe(II)"
    },
    {
        "Reactivo": "NH4OH o NaOH",
        "Función": "Agente precipitante"
    },
    {
        "Reactivo": "Agua desionizada",
        "Función": "Disolución y lavado"
    },
    {
        "Reactivo": "Etanol",
        "Función": "Lavado opcional"
    }
])

st.dataframe(
    reactivos,
    use_container_width=True,
    hide_index=True
)

st.subheader("Materiales y equipo")

equipo = pd.DataFrame([
    {
        "Equipo": "Balanza analítica",
        "Uso": "Pesaje de precursores"
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
        "Uso": "Agitación y temperatura"
    },
    {
        "Equipo": "Barras magnéticas de PTFE",
        "Uso": "Agitación"
    },
    {
        "Equipo": "pH-metro",
        "Uso": "Control del pH"
    },
    {
        "Equipo": "Jeringa o bureta",
        "Uso": "Adición controlada de la base"
    },
    {
        "Equipo": "Termómetro",
        "Uso": "Control de temperatura"
    },
    {
        "Equipo": "Centrífuga o filtración al vacío",
        "Uso": "Separación del sólido"
    },
    {
        "Equipo": "Papel filtro y embudo Büchner",
        "Uso": "Recuperación del material"
    },
    {
        "Equipo": "Estufa u horno de vacío",
        "Uso": "Secado"
    },
    {
        "Equipo": "Desecador",
        "Uso": "Almacenamiento"
    },
    {
        "Equipo": "Campana de extracción",
        "Uso": "Seguridad química"
    },
    {
        "Equipo": "Bata, guantes y gafas",
        "Uso": "Protección personal"
    }
])

st.dataframe(
    equipo,
    use_container_width=True,
    hide_index=True
)

st.subheader("Procedimiento paso a paso")

pasos = [
    "Revisar las hojas de seguridad y preparar el área de trabajo con bata, guantes, gafas y campana de extracción.",

    "Definir la escala de síntesis y calcular las cantidades de FeCl3·6H2O y FeCl2·4H2O con base en un protocolo validado.",

    "Pesar las sales de hierro utilizando una balanza analítica.",

    "Disolver por separado los precursores en agua desionizada.",

    "Preparar la solución de NH4OH o NaOH en un recipiente independiente.",

    "Colocar la solución de hierro bajo agitación magnética.",

    "Agregar lentamente el agente precipitante mientras se controla el pH.",

    "Mantener la agitación durante el tiempo de envejecimiento establecido en el protocolo.",

    "Separar las nanopartículas mediante centrifugación o filtración al vacío.",

    "Lavar el sólido varias veces con agua desionizada.",

    "Realizar un lavado opcional con etanol si el protocolo lo indica.",

    "Secar el sólido utilizando una estufa o un horno de vacío.",

    "Guardar la muestra en un desecador y registrar las condiciones experimentales.",

    "Caracterizar el material mediante difracción de rayos X, microscopía y magnetometría.",

    "Comparar las propiedades medidas con las propiedades objetivo."
]

for numero, paso in enumerate(pasos, start=1):
    st.markdown(f"**Paso {numero}.** {paso}")

st.warning(
    """
    Las cantidades exactas, concentraciones, temperaturas y tiempos deben
    tomarse de un protocolo validado en la literatura y aprobado por el
    laboratorio. Esta aplicación es una demostración educativa.
    """
)

# ============================================================
# COSTOS
# ============================================================

st.header("7. Costo preliminar")

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
    "Costo preliminar",
    f"${costo_total:,.2f} MXN"
)

st.caption(
    "Los precios son ilustrativos y deben sustituirse por cotizaciones reales."
)
