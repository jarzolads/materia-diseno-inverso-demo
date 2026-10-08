import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from google import genai
from google.genai import types
from mp_api.client import MPRester


st.set_page_config(
    page_title="Matéria",
    page_icon="🔬",
    layout="wide",
)


# =========================================================
# CONFIGURACIÓN
# =========================================================

MODELO_GEMINI = "gemini-3.8-flash"

EJEMPLO = (
    "Busca óxidos de hierro con magnetización calculada "
    "por volumen mínima de 0.02 μB/Å³ y energía sobre la "
    "envolvente máxima de 0.05 eV/átomo."
)


def leer_secreto(nombre, valor_default=""):
    try:
        return st.secrets.get(nombre, valor_default)
    except FileNotFoundError:
        return valor_default


# =========================================================
# GLOSARIO
# =========================================================

GLOSARIO = {
    "Magnetización por volumen": {
        "unidad": "μB/Å³",
        "definicion": (
            "Es el momento magnético calculado por unidad de volumen "
            "del cristal."
        ),
        "importancia": (
            "Ayuda a identificar materiales con comportamiento magnético "
            "calculado potencialmente interesante."
        ),
        "limitacion": (
            "No equivale al SAR y no predice directamente cuánto calentará "
            "una nanopartícula."
        ),
    },
    "Energía sobre la envolvente": {
        "unidad": "eV/átomo",
        "definicion": (
            "Es la distancia energética entre el material y la combinación "
            "de fases más estable calculada para su composición."
        ),
        "importancia": (
            "Un valor menor indica mayor estabilidad termodinámica calculada "
            "respecto a las fases de referencia."
        ),
        "limitacion": (
            "No garantiza que el material pueda sintetizarse fácilmente."
        ),
    },
    "Ordenamiento magnético": {
        "unidad": "clasificación",
        "definicion": (
            "Describe la configuración calculada de los momentos magnéticos, "
            "por ejemplo ferromagnética o antiferromagnética."
        ),
        "importancia": (
            "Ayuda a describir el comportamiento magnético calculado "
            "del material."
        ),
        "limitacion": (
            "No demuestra el comportamiento de una nanopartícula real."
        ),
    },
    "Densidad": {
        "unidad": "g/cm³",
        "definicion": "Es la masa del material por unidad de volumen.",
        "importancia": (
            "Ayuda a comparar materiales y estimar cantidades requeridas."
        ),
        "limitacion": (
            "No es una medida directa del desempeño en hipertermia."
        ),
    },
}


# =========================================================
# SÍNTESIS Y COSTOS
# =========================================================

PASOS_SINTESIS = [
    "Seleccionar precursores de hierro(II) y hierro(III).",
    "Preparar una disolución con concentraciones conocidas.",
    "Agregar una base bajo agitación y pH controlado.",
    "Separar y lavar el precipitado.",
    "Secar el sólido obtenido.",
    "Caracterizar fase, tamaño y propiedades magnéticas.",
]

COSTOS = pd.DataFrame([
    {
        "Material": "Precursor de Fe(III)",
        "Cantidad_g": 0.80,
        "Precio_paquete_MXN": 850.0,
        "Paquete_g": 500.0,
    },
    {
        "Material": "Precursor de Fe(II)",
        "Cantidad_g": 0.40,
        "Precio_paquete_MXN": 1200.0,
        "Paquete_g": 500.0,
    },
    {
        "Material": "Base",
        "Cantidad_g": 0.60,
        "Precio_paquete_MXN": 300.0,
        "Paquete_g": 1000.0,
    },
    {
        "Material": "Agua desionizada",
        "Cantidad_g": 20.0,
        "Precio_paquete_MXN": 80.0,
        "Paquete_g": 1000.0,
    },
])


def calcular_costos():
    tabla = COSTOS.copy()

    tabla["Costo_MXN"] = (
        tabla["Cantidad_g"]
        * tabla["Precio_paquete_MXN"]
        / tabla["Paquete_g"]
    )

    precursores = tabla["Costo_MXN"].sum()
    energia = 25.0
    consumibles = 50.0
    total = precursores + energia + consumibles

    return tabla, precursores, energia, consumibles, total


# =========================================================
# MATERIALS PROJECT
# =========================================================


def cargar_materiales():
    api_key = leer_secreto("MP_API_KEY")

    if not api_key:
        raise ValueError(
            "Falta MP_API_KEY en los secretos de Streamlit."
        )

    campos = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "density",
        "energy_above_hull",
        "ordering",
        "total_magnetization_normalized_vol",
        "symmetry",
    ]

    with MPRester(
        api_key,
        use_document_model=False,
    ) as mpr:

        documentos = mpr.materials.summary.search(
            chemsys=["Fe-O"],
            deprecated=False,
            fields=campos,
        )

    filas = []

    for documento in documentos:

        ordenamiento = documento.get("ordering")

        if hasattr(ordenamiento, "value"):
            ordenamiento = ordenamiento.value

        simetria = documento.get("symmetry") or {}

        filas.append({
            "material_id": str(
                documento.get("material_id")
            ),
            "formula": documento.get(
                "formula_pretty"
            ),
            "sistema": documento.get(
                "chemsys"
            ),
            "densidad": documento.get(
                "density"
            ),
            "E_hull": documento.get(
                "energy_above_hull"
            ),
            "M_vol": documento.get(
                "total_magnetization_normalized_vol"
            ),
            "ordenamiento": ordenamiento,
            "grupo_espacial": simetria.get(
                "symbol"
            ),
        })

    datos = pd.DataFrame(filas)

    for columna in [
        "densidad",
        "E_hull",
        "M_vol",
    ]:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    return datos


# =========================================================
# BÚSQUEDA INVERSA
# =========================================================


def buscar_materiales(
    datos,
    magnetizacion_minima,
    hull_maximo,
    densidad_maxima=None,
):
    resultado = datos.copy()

    estados = []
    motivos = []
    distancias = []

    for _, fila in resultado.iterrows():

        m_vol = fila["M_vol"]
        e_hull = fila["E_hull"]
        densidad = fila["densidad"]

        razones = []
        distancia = 0.0

        if pd.isna(m_vol):
            estados.append("Sin datos suficientes")
            motivos.append(
                "Falta magnetización calculada"
            )
            distancias.append(np.nan)
            continue

        if pd.isna(e_hull):
            estados.append("Sin datos suficientes")
            motivos.append(
                "Falta energía sobre la envolvente"
            )
            distancias.append(np.nan)
            continue

        if e_hull > hull_maximo:
            razones.append(
                "Supera E_hull máximo"
            )

        if m_vol < magnetizacion_minima:
            razones.append(
                "Magnetización inferior al objetivo"
            )

            distancia += (
                magnetizacion_minima - m_vol
            ) / magnetizacion_minima

        if (
            densidad_maxima is not None
            and not pd.isna(densidad)
            and densidad > densidad_maxima
        ):
            razones.append(
                "Densidad superior al objetivo"
            )

            distancia += (
                densidad - densidad_maxima
            ) / densidad_maxima

        if e_hull <= hull_maximo and not razones:
            estado = "Cumple objetivos"
        elif e_hull > hull_maximo:
            estado = "Excluido"
        else:
            estado = "Aproximado"

        estados.append(estado)
        motivos.append("; ".join(razones))
        distancias.append(distancia)

    resultado["estado"] = estados
    resultado["motivos"] = motivos
    resultado["distancia"] = distancias

    orden = {
        "Cumple objetivos": 0,
        "Aproximado": 1,
        "Sin datos suficientes": 2,
        "Excluido": 3,
    }

    resultado["_orden"] = resultado["estado"].map(orden)

    resultado = resultado.sort_values(
        ["_orden", "distancia"],
        na_position="last",
    )

    return resultado.drop(
        columns="_orden"
    ).reset_index(drop=True)


# =========================================================
# HERRAMIENTA DE GEMINI
# =========================================================


DECLARACION = types.FunctionDeclaration(
    name="buscar_materiales",
    description=(
        "Busca materiales de Fe-O según una magnetización mínima, "
        "una energía máxima sobre la envolvente y una densidad "
        "máxima opcional."
    ),
    parameters={
        "type": "object",
        "properties": {
            "magnetizacion_minima": {
                "type": "number",
                "description": "Valor mínimo en μB/Å³.",
            },
            "hull_maximo": {
                "type": "number",
                "description": "Valor máximo en eV/átomo.",
            },
            "densidad_maxima": {
                "type": "number",
                "description": (
                    "Valor máximo en g/cm³. "
                    "Usa -1 si el usuario no solicita densidad."
                ),
            },
        },
        "required": [
            "magnetizacion_minima",
            "hull_maximo",
            "densidad_maxima",
        ],
    },
)


INSTRUCCIONES = """
Eres Matéria, un agente educativo de diseño inverso de materiales.

El usuario expresa propiedades deseadas y tú las conviertes
en argumentos para la herramienta buscar_materiales.

Reglas:

- Trabaja únicamente con materiales del sistema Fe-O.
- No inventes propiedades.
- La magnetización está en μB/Å³.
- La energía sobre la envolvente está en eV/átomo.
- La densidad está en g/cm³.
- Si el usuario no solicita densidad, usa -1.
- Si pide más magnetización, aumenta el mínimo.
- Si pide mayor estabilidad, reduce el valor máximo de E_hull.
- No afirmes que un material tendrá alto SAR.
- No afirmes que las propiedades calculadas demuestran biocompatibilidad.
- Explica la solicitud en español.
- Menciona como máximo tres material_id.
"""


def ejecutar_agente(pregunta, datos):

    api_key = leer_secreto("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Falta GEMINI_API_KEY en los secretos."
        )

    cliente = genai.Client(
        api_key=api_key
    )

    configuracion = types.GenerateContentConfig(
        system_instruction=INSTRUCCIONES,
        tools=[
            types.Tool(
                function_declarations=[
                    DECLARACION
                ]
            )
        ],
        automatic_function_calling=(
            types.AutomaticFunctionCallingConfig(
                disable=True
            )
        ),
    )

    respuesta = cliente.models.generate_content(
        model=leer_secreto(
            "GEMINI_MODEL",
            MODELO_GEMINI,
        ),
        contents=pregunta,
        config=configuracion,
    )

    llamadas = []

    if respuesta.candidates:

        contenido = respuesta.candidates[0].content

        if contenido and contenido.parts:
            llamadas = [
                parte.function_call
                for parte in contenido.parts
                if parte.function_call is not None
            ]

    if not llamadas:
        return (
            respuesta.text
            or "Gemini no generó una solicitud estructurada.",
            None,
        )

    llamada = llamadas[0]
    argumentos = dict(llamada.args)

    magnetizacion = float(
        argumentos["magnetizacion_minima"]
    )

    hull = float(
        argumentos["hull_maximo"]
    )

    densidad = float(
        argumentos["densidad_maxima"]
    )

    if densidad < 0:
        densidad = None

    resultado = buscar_materiales(
        datos,
        magnetizacion_minima=magnetizacion,
        hull_maximo=hull,
        densidad_maxima=densidad,
    )

    resumen = {
        "total_registros": len(resultado),
        "cumplen": int(
            (
                resultado["estado"]
                == "Cumple objetivos"
            ).sum()
        ),
        "aproximados": int(
            (
                resultado["estado"]
                == "Aproximado"
            ).sum()
        ),
        "candidatos": json.loads(
            resultado.head(5).to_json(
                orient="records",
                double_precision=8,
            )
        ),
    }

    respuesta_funcion = types.Part(
        function_response=types.FunctionResponse(
            name="buscar_materiales",
            response=resumen,
        )
    )

    respuesta_final = cliente.models.generate_content(
        model=leer_secreto(
            "GEMINI_MODEL",
            MODELO_GEMINI,
        ),
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=pregunta
                    )
                ],
            ),
            respuesta.candidates[0].content,
            types.Content(
                role="user",
                parts=[
                    respuesta_funcion
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCCIONES,
        ),
    )

    return (
        respuesta_final.text
        or "La búsqueda fue ejecutada.",
        resultado,
    )


# =========================================================
# ESTADO
# =========================================================


if "datos" not in st.session_state:
    st.session_state.datos = None

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "historial" not in st.session_state:
    st.session_state.historial = []


# =========================================================
# INTERFAZ PRINCIPAL
# =========================================================


st.title("🔬 Matéria")

st.subheader(
    "Demostración de diseño inverso de materiales"
)

st.write(
    "El usuario comienza con las propiedades que desea. "
    "Gemini interpreta la solicitud y Python busca materiales "
    "que se aproximen a esos objetivos."
)

st.info(
    "Ejemplo: busca óxidos de hierro con magnetización "
    "mínima de 0.02 μB/Å³ y E_hull máximo de 0.05 eV/átomo."
)


col1, col2 = st.columns(2)

if col1.button(
    "Cargar datos de Materials Project",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Consultando Materials Project..."
        ):
            st.session_state.datos = (
                cargar_materiales()
            )

        st.success(
            f"Se cargaron "
            f"{len(st.session_state.datos)} registros."
        )

    except Exception as error:

        st.error(
            f"No se pudieron cargar los datos: "
            f"{type(error).__name__}: {error}"
        )


if col2.button(
    "Probar ejemplo",
    use_container_width=True,
):

    if st.session_state.datos is None:

        st.warning(
            "Primero carga los datos."
        )

    else:

        try:

            with st.spinner(
                "Gemini está interpretando la solicitud..."
            ):
                texto, resultado = ejecutar_agente(
                    EJEMPLO,
                    st.session_state.datos,
                )

            st.session_state.historial.append(
                {
                    "usuario": EJEMPLO,
                    "agente": texto,
                }
            )

            st.session_state.resultado = resultado

        except Exception as error:

            st.error(
                f"No se pudo ejecutar el agente: "
                f"{type(error).__name__}: {error}"
            )


st.divider()

pregunta = st.chat_input(
    "Escribe las propiedades deseadas"
)

if pregunta:

    if st.session_state.datos is None:

        st.warning(
            "Primero carga los datos."
        )

    else:

        try:

            with st.spinner(
                "Gemini está interpretando tu solicitud..."
            ):

                texto, resultado = ejecutar_agente(
                    pregunta,
                    st.session_state.datos,
                )

            st.session_state.historial.append(
                {
                    "usuario": pregunta,
                    "agente": texto,
                }
            )

            st.session_state.resultado = resultado

        except Exception as error:

            st.error(
                f"No se pudo ejecutar la búsqueda: "
                f"{type(error).__name__}: {error}"
            )


# =========================================================
# CONVERSACIÓN
# =========================================================


if st.session_state.historial:

    st.subheader("Conversación")

    for intercambio in st.session_state.historial:

        with st.chat_message("user"):
            st.write(
                intercambio["usuario"]
            )

        with st.chat_message("assistant"):
            st.write(
                intercambio["agente"]
            )


# =========================================================
# RESULTADOS
# =========================================================


if st.session_state.resultado is not None:

    resultado = st.session_state.resultado

    st.subheader("Candidatos encontrados")

    completos = resultado[
        resultado["estado"]
        == "Cumple objetivos"
    ]

    aproximados = resultado[
        resultado["estado"]
        == "Aproximado"
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Cumplen objetivos",
        len(completos),
    )

    c2.metric(
        "Candidatos aproximados",
        len(aproximados),
    )

    st.dataframe(
        resultado[
            [
                "formula",
                "material_id",
                "M_vol",
                "E_hull",
                "densidad",
                "ordenamiento",
                "estado",
                "motivos",
            ]
        ].head(10),
        hide_index=True,
        use_container_width=True,
    )

    graficables = resultado.dropna(
        subset=["M_vol", "E_hull"]
    )

    if not graficables.empty:

        figura = px.scatter(
            graficables,
            x="E_hull",
            y="M_vol",
            color="estado",
            hover_name="formula",
            hover_data=[
                "material_id",
                "densidad",
                "ordenamiento",
            ],
            labels={
                "E_hull": (
                    "Energía sobre la envolvente "
                    "(eV/átomo)"
                ),
                "M_vol": (
                    "Magnetización (μB/Å³)"
                ),
            },
        )

        st.plotly_chart(
            figura,
            use_container_width=True,
        )


# =========================================================
# GLOSARIO
# =========================================================


st.divider()

st.subheader(
    "Conceptos científicos"
)

for nombre, informacion in GLOSARIO.items():

    with st.expander(
        f"{nombre} ({informacion['unidad']})"
    ):

        st.write(
            informacion["definicion"]
        )

        st.markdown(
            f"**Por qué importa:** "
            f"{informacion['importancia']}"
        )

        st.markdown(
            f"**Limitación:** "
            f"{informacion['limitacion']}"
        )


# =========================================================
# SÍNTESIS Y COSTOS
# =========================================================


st.divider()

st.subheader(
    "Estrategia de síntesis y costo preliminar"
)

st.write(
    "Esta sección es educativa. Los precios son ilustrativos "
    "y deben sustituirse por cotizaciones reales."
)

st.markdown(
    "**Material de referencia:** Fe₃O₄"
)

st.markdown(
    "**Estrategia:** coprecipitación"
)

st.caption(
    "La estrategia es una plantilla educativa y no sustituye "
    "un protocolo bibliográfico validado."
)

for paso in PASOS_SINTESIS:
    st.write(f"• {paso}")

tabla_costos, precursores, energia, consumibles, total = (
    calcular_costos()
)

st.dataframe(
    tabla_costos,
    hide_index=True,
    use_container_width=True,
)

a, b, c = st.columns(3)

a.metric(
    "Precursores",
    f"${precursores:.2f} MXN",
)

b.metric(
    "Energía y consumibles",
    f"${energia + consumibles:.2f} MXN",
)

c.metric(
    "Total preliminar",
    f"${total:.2f} MXN",
)

st.warning(
    "Los costos mostrados son ejemplos didácticos. "
    "No representan una cotización comercial."
)
