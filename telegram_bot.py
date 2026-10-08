import os
import json
import asyncio
import numpy as np
import pandas as pd

from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
)

MP_API_KEY = os.getenv(
    "MP_API_KEY",
    ""
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# ============================================================
# FAMILIAS DE MATERIALES
# ============================================================

FAMILIAS = {
    "Óxidos metálicos": [
        "Fe-O",
        "Co-O",
        "Ni-O",
        "Mn-O",
        "Ti-O",
        "Zn-O",
        "Al-O",
        "Zr-O"
    ],

    "Ferritas": [
        "Fe-Co-O",
        "Fe-Ni-O",
        "Fe-Mn-O",
        "Fe-Zn-O",
        "Fe-Cu-O"
    ],

    "Aleaciones magnéticas": [
        "Fe-Co",
        "Fe-Ni",
        "Co-Ni",
        "Fe-Mn",
        "Co-Mn"
    ],

    "Materiales para baterías": [
        "Li-Fe-O",
        "Li-Co-O",
        "Li-Ni-O",
        "Li-Mn-O",
        "Na-Fe-O",
        "Na-Mn-O"
    ],

    "Cerámicos": [
        "Ba-Ti-O",
        "Sr-Ti-O",
        "Al-O",
        "Zr-O",
        "Ca-Ti-O"
    ]
}

# ============================================================
# UTILIDADES
# ============================================================

def partir_mensaje(texto, limite=3900):
    """
    Telegram tiene un límite aproximado de 4096 caracteres por mensaje.
    """

    partes = []

    while len(texto) > limite:
        posicion = texto.rfind("\n", 0, limite)

        if posicion == -1:
            posicion = limite

        partes.append(texto[:posicion])
        texto = texto[posicion:]

    if texto:
        partes.append(texto)

    return partes


def elemento_valor(elemento, propiedad, defecto=0.0):
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
            elemento_valor(elemento, "Z")
            for elemento in elementos
        ])

        masas = np.array([
            elemento_valor(elemento, "atomic_mass")
            for elemento in elementos
        ])

        electronegatividades = np.array([
            elemento_valor(elemento, "X")
            for elemento in elementos
        ])

        radios = np.array([
            elemento_valor(elemento, "atomic_radius")
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


# ============================================================
# GEMINI
# ============================================================

def interpretar_prompt(prompt):
    """
    Convierte el prompt del usuario en criterios de búsqueda.
    """

    valores_defecto = {
        "familia": "Ferritas",
        "elemento_a": "Co",
        "elemento_b": "Ni",
        "elemento_base": "Fe",
        "anion": "O",
        "sistema_quimico": "",
        "magnetizacion_minima": 0.0,
        "energia_maxima": 0.10,
        "densidad_maxima": 0.0
    }

    if not GEMINI_API_KEY:
        return valores_defecto

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        instruccion = f"""
        Interpreta este objetivo de materiales:

        {prompt}

        Devuelve únicamente JSON válido con esta estructura:

        {{
          "familia": "Ferritas",
          "elemento_a": "Co",
          "elemento_b": "Ni",
          "elemento_base": "Fe",
          "anion": "O",
          "sistema_quimico": "",
          "magnetizacion_minima": 0.0,
          "energia_maxima": 0.10,
          "densidad_maxima": 0.0
        }}

        La familia debe ser una de estas:

        - Óxidos metálicos
        - Ferritas
        - Aleaciones magnéticas
        - Materiales para baterías
        - Cerámicos

        Si un dato no aparece, utiliza el valor predeterminado.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instruccion
        )

        texto = respuesta.text.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        inicio = texto.find("{")
        final = texto.rfind("}")

        if inicio == -1 or final == -1:
            return valores_defecto

        valores = json.loads(
            texto[inicio:final + 1]
        )

        for clave, valor in valores_defecto.items():
            if clave not in valores:
                valores[clave] = valor

        if valores["familia"] not in FAMILIAS:
            valores["familia"] = "Ferritas"

        return valores

    except Exception:
        return valores_defecto


def interpretar_resultados(prompt, resultados):
    if not GEMINI_API_KEY:
        return (
            "Gemini no está disponible. Se muestran los resultados "
            "obtenidos de Materials Project."
        )

    try:
        from google import genai

        cliente = genai.Client(
            api_key=GEMINI_API_KEY
        )

        datos = resultados.to_dict(
            orient="records"
        )

        instruccion = f"""
        Actúa como un agente educativo de materiales.

        Objetivo:
        {prompt}

        Candidatos:
        {json.dumps(datos, ensure_ascii=False, default=str)}

        Responde en español con estas secciones:

        1. Interpretación del objetivo.
        2. Comparación de los candidatos.
        3. Candidato recomendado.
        4. Limitaciones de los datos.
        5. Rutas de síntesis que podrían explorarse.
        6. Precursores químicos posibles y función de cada uno.
        7. Material y equipo de laboratorio.
        8. Caracterización recomendada.
        9. Costo preliminar.
        10. Proveedores potenciales y SDS.
        11. Laboratorios de la BUAP que podrían ser pertinentes.

        Utiliza lenguaje claro. No inventes enlaces, disponibilidad,
        responsables ni equipos de la BUAP. Aclara que todo debe verificarse.
        """

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=instruccion
        )

        return respuesta.text

    except Exception as error:
        return f"No fue posible interpretar los resultados: {error}"


# ============================================================
# MATERIALS PROJECT
# ============================================================

def consultar_materials_project(
    sistemas,
    energia_maxima=0.10,
    magnetizacion_minima=0.0,
    limite=50
):
    """
    Consulta materiales reales de Materials Project.
    """

    if not MP_API_KEY:
        return pd.DataFrame()

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

    registros = []
    ids = set()

    with MPRester(MP_API_KEY) as mpr:

        for sistema in sistemas:

            try:
                documentos = mpr.materials.summary.search(
                    chemsys=sistema,
                    fields=campos,
                    num_chunks=1
                )

            except Exception:
                campos_basicos = [
                    "material_id",
                    "formula_pretty",
                    "chemsys",
                    "density",
                    "energy_above_hull",
                    "ordering",
                    "symmetry"
                ]

                documentos = mpr.materials.summary.search(
                    chemsys=sistema,
                    fields=campos_basicos,
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

                if material_id in ids:
                    continue

                energia = getattr(
                    documento,
                    "energy_above_hull",
                    np.nan
                )

                if energia is None:
                    continue

                if float(energia) > energia_maxima:
                    continue

                magnetizacion = getattr(
                    documento,
                    "total_magnetization_normalized_vol",
                    np.nan
                )

                if (
                    magnetizacion_minima > 0
                    and (
                        magnetizacion is None
                        or pd.isna(magnetizacion)
                        or float(magnetizacion)
                        < magnetizacion_minima
                    )
                ):
                    continue

                registros.append({
                    "material_id": material_id,
                    "formula_pretty": getattr(
                        documento,
                        "formula_pretty",
                        ""
                    ),
                    "chemsys": getattr(
                        documento,
                        "chemsys",
                        ""
                    ),
                    "density": getattr(
                        documento,
                        "density",
                        np.nan
                    ),
                    "energy_above_hull": energia,
                    "ordering": str(
                        getattr(
                            documento,
                            "ordering",
                            "No disponible"
                        )
                    ),
                    "magnetization": magnetizacion,
                    "symmetry": str(
                        getattr(
                            documento,
                            "symmetry",
                            "No disponible"
                        )
                    )
                })

                ids.add(material_id)

                if len(registros) >= limite:
                    break

            if len(registros) >= limite:
                break

    return pd.DataFrame(registros)


# ============================================================
# DISEÑO GENERATIVO
# ============================================================

def construir_dataset_ml(documentos):
    registros = []

    for documento in documentos:

        formula = getattr(
            documento,
            "formula_pretty",
            None
        )

        if not formula:
            continue

        features = caracteristicas_composicion(
            formula
        )

        if any(pd.isna(features)):
            continue

        density = getattr(
            documento,
            "density",
            np.nan
        )

        energy = getattr(
            documento,
            "energy_above_hull",
            np.nan
        )

        if pd.isna(density) or pd.isna(energy):
            continue

        registros.append({
            "formula": formula,
            "density": float(density),
            "energy_above_hull": float(energy),
            "features": features
        })

    return pd.DataFrame(registros)


def generar_formulaciones(
    familia,
    elemento_a,
    elemento_b,
    elemento_base,
    anion
):
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

            partes = []

            if fraccion_a > 0:
                partes.append(
                    f"{elemento_a}{fraccion_a:g}"
                )

            if fraccion_b > 0:
                partes.append(
                    f"{elemento_b}{fraccion_b:g}"
                )

            if familia == "Ferritas":
                formula = (
                    "".join(partes)
                    + f"{elemento_base}2{anion}4"
                )

            elif familia == "Óxidos metálicos":
                formula = (
                    "".join(partes)
                    + f"{anion}3"
                )

            elif familia == "Aleaciones magnéticas":
                formula = "".join(partes)

            elif familia == "Materiales para baterías":
                formula = (
                    "Li"
                    + "".join(partes)
                    + f"{anion}2"
                )

            else:
                formula = (
                    "".join(partes)
                    + f"{elemento_base}{anion}3"
                )

            formula = Composition(
                formula
            ).reduced_formula

            if formula not in formulaciones:
                formulaciones.append(formula)

        except Exception:
            continue

    return formulaciones


def entrenar_modelos(datos):
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

    modelos["densidad"] = modelo_densidad
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

    modelo_ehull = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    modelo_ehull.fit(
        x_train,
        y_train
    )

    predicciones = modelo_ehull.predict(
        x_test
    )

    modelos["ehull"] = modelo_ehull
    modelos["mae_ehull"] = mean_absolute_error(
        y_test,
        predicciones
    )
    modelos["r2_ehull"] = r2_score(
        y_test,
        predicciones
    )

    return modelos


def evaluar_formulaciones(
    formulaciones,
    modelos,
    formulas_conocidas
):
    registros = []

    for formula in formulaciones:

        features = caracteristicas_composicion(
            formula
        )

        if any(pd.isna(features)):
            continue

        vector = np.array(
            features
        ).reshape(1, -1)

        density = modelos["densidad"].predict(
            vector
        )[0]

        ehull = modelos["ehull"].predict(
            vector
        )[0]

        registros.append({
            "Fórmula": formula,
            "Densidad predicha": density,
            "E_hull predicho": ehull,
            "Estado": (
                "Conocida"
                if formula in formulas_conocidas
                else "Hipotética"
            )
        })

    return pd.DataFrame(registros)


# ============================================================
# COMANDOS DE TELEGRAM
# ============================================================

async def comando_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    mensaje = """
Hola. Soy Matéria, un agente para el diseño inverso y generativo de materiales.

Puedes utilizar:

/buscar seguido de un objetivo para consultar Materials Project.

/generar seguido de un objetivo para proponer formulaciones nuevas.

/ayuda para ver ejemplos.

Ejemplo:

/buscar busca ferritas estables con alta magnetización

/generar diseña una ferrita hipotética basada en Co, Ni, Fe y O
"""

    await update.message.reply_text(
        mensaje
    )


async def comando_ayuda(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    mensaje = """
Ejemplos de uso:

/buscar
Busca materiales magnéticos estables basados en hierro y oxígeno.

/buscar
Busca candidatos con baja energía sobre el envolvente y alta magnetización.

/generar
Genera una ferrita hipotética basada en Co, Ni, Fe y O.

/generar
Diseña una formulación para materiales de batería basada en Li, Ni, Mn y O.

El bot devuelve candidatos, predicciones, rutas de síntesis y caracterización.
"""

    await update.message.reply_text(
        mensaje
    )


async def comando_buscar(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    prompt = " ".join(
        context.args
    ).strip()

    if not prompt:
        await update.message.reply_text(
            "Escribe un objetivo después de /buscar."
        )
        return

    await update.message.reply_text(
        "Estoy interpretando tu objetivo y consultando Materials Project..."
    )

    try:
        valores = await asyncio.to_thread(
            interpretar_prompt,
            prompt
        )

        familia = valores["familia"]
        sistemas = FAMILIAS[familia]

        resultados = await asyncio.to_thread(
            consultar_materials_project,
            sistemas,
            valores["energia_maxima"],
            valores["magnetizacion_minima"],
            20
        )

        if resultados.empty:
            await update.message.reply_text(
                "No encontré candidatos con esos criterios."
            )
            return

        mejores = resultados.head(5)

        resumen = mejores.to_string(
            index=False
        )

        respuesta = (
            f"Familia consultada: {familia}\n\n"
            f"Cinco mejores candidatos:\n\n"
            f"{resumen}\n\n"
            f"Estoy preparando la interpretación científica..."
        )

        await update.message.reply_text(
            "\n".join(
                partir_mensaje(respuesta)
            )
        )

        interpretacion = await asyncio.to_thread(
            interpretar_resultados,
            prompt,
            mejores
        )

        for parte in partir_mensaje(
            interpretacion
        ):
            await update.message.reply_text(
                parte
            )

    except Exception as error:
        await update.message.reply_text(
            f"Ocurrió un error durante la búsqueda:\n{error}"
        )


async def comando_generar(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    prompt = " ".join(
        context.args
    ).strip()

    if not prompt:
        await update.message.reply_text(
            "Escribe un objetivo después de /generar."
        )
        return

    await update.message.reply_text(
        "Estoy interpretando el objetivo y preparando el diseño generativo..."
    )

    try:
        valores = await asyncio.to_thread(
            interpretar_prompt,
            prompt
        )

        familia = valores["familia"]

        sistemas = FAMILIAS[familia]

        documentos = await asyncio.to_thread(
            consultar_materials_project,
            sistemas,
            0.50,
            0.0,
            100
        )

        if documentos.empty:
            await update.message.reply_text(
                "No fue posible obtener datos para entrenar el modelo."
            )
            return

        documentos_ml = await asyncio.to_thread(
            construir_dataset_ml_from_results,
            documentos
        )

        if len(documentos_ml) < 15:
            await update.message.reply_text(
                "El conjunto de entrenamiento es demasiado pequeño."
            )
            return

        modelos = await asyncio.to_thread(
            entrenar_modelos,
            documentos_ml
        )

        formulaciones = generar_formulaciones(
            familia=familia,
            elemento_a=valores["elemento_a"],
            elemento_b=valores["elemento_b"],
            elemento_base=valores["elemento_base"],
            anion=valores["anion"]
        )

        formulas_conocidas = set(
            documentos_ml["formula"].astype(str)
        )

        resultados = evaluar_formulaciones(
            formulaciones,
            modelos,
            formulas_conocidas
        )

        if resultados.empty:
            await update.message.reply_text(
                "No fue posible generar formulaciones válidas."
            )
            return

        resultados = resultados.sort_values(
            by="E_hull predicho"
        ).head(5)

        tabla = resultados.to_string(
            index=False
        )

        mensaje = (
            f"Familia: {familia}\n"
            f"Elementos: {valores['elemento_a']}, "
            f"{valores['elemento_b']}, "
            f"{valores['elemento_base']} y "
            f"{valores['anion']}\n\n"
            f"Formulaciones propuestas:\n\n"
            f"{tabla}\n\n"
            f"Preparando interpretación y rutas de síntesis..."
        )

        for parte in partir_mensaje(mensaje):
            await update.message.reply_text(
                parte
            )

        interpretacion = await asyncio.to_thread(
            interpretar_resultados,
            prompt,
            resultados
        )

        for parte in partir_mensaje(
            interpretacion
        ):
            await update.message.reply_text(
                parte
            )

    except Exception as error:
        await update.message.reply_text(
            f"Ocurrió un error durante el diseño generativo:\n{error}"
        )


async def recibir_texto(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        """
Utiliza uno de estos comandos:

/buscar tu objetivo

/generar tu objetivo

/ayuda
"""
    )


# ============================================================
# CONVERSIÓN DE RESULTADOS
# ============================================================

def construir_dataset_ml_from_results(resultados):
    registros = []

    for _, fila in resultados.iterrows():

        formula = fila["formula_pretty"]

        features = caracteristicas_composicion(
            formula
        )

        if any(pd.isna(features)):
            continue

        registros.append({
            "formula": formula,
            "density": float(
                fila["density"]
            ),
            "energy_above_hull": float(
                fila["energy_above_hull"]
            ),
            "features": features
        })

    return pd.DataFrame(registros)


# ============================================================
# INICIO DEL BOT
# ============================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "No existe TELEGRAM_BOT_TOKEN."
        )

    if not MP_API_KEY:
        raise RuntimeError(
            "No existe MP_API_KEY."
        )

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "No existe GEMINI_API_KEY."
        )

    aplicacion = (
        Application
        .builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    aplicacion.add_handler(
        CommandHandler(
            "start",
            comando_start
        )
    )

    aplicacion.add_handler(
        CommandHandler(
            "ayuda",
            comando_ayuda
        )
    )

    aplicacion.add_handler(
        CommandHandler(
            "buscar",
            comando_buscar
        )
    )

    aplicacion.add_handler(
        CommandHandler(
            "generar",
            comando_generar
        )
    )

    aplicacion.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            recibir_texto
        )
    )

    print("Matéria está ejecutándose en Telegram.")

    aplicacion.run_polling()


if __name__ == "__main__":
    main()
