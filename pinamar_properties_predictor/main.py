from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel, Field
from typing import Literal

# Inicializamos la API
app = FastAPI(
    title="Tasador Inmobiliario Pinamar API",
    description="API para predecir precios de propiedades usando Gradient Boosting",
    version="1.0.0"
)

print("Cargando preprocesador y modelo...")
# Cargamos el preprocesador y el modelo ganador
try:
    preprocessor = joblib.load('models/trained_models/preprocesador_pinamar.pkl')
    model_gb_log = joblib.load('models/trained_models/modelo_gb_log_pinamar.joblib')
    print("✅ Modelos cargados correctamente en memoria.")
except Exception as e:
    print(f"❌ Error al cargar los modelos: {e}")


class CasaInput(BaseModel):
    # 1. Numéricas (¡Eliminamos 'ambientes_extra' de aquí!)
    m2: float = Field(..., description="Metros cuadrados totales")
    m2_construidos_final: float = Field(..., description="Metros cuadrados construidos")
    ambientes: int = Field(..., description="Cantidad total de ambientes")
    dormitorios: int = Field(..., description="Cantidad de dormitorios")
    baños: int = Field(..., description="Cantidad de baños")
    cocheras: int = Field(..., description="Cantidad de cocheras")
    nivel_pileta: int = Field(..., description="Nivel o tamaño de la pileta (0 si no tiene)")
    cantidad_plantas: int = Field(..., description="Cantidad de pisos/plantas")

    # 2. Categóricas CON OPCIONES DESPLEGABLES (Dropdowns)
    # *Nota: Edita estas listas con los valores exactos que existan en tus datos limpios*
    barrio_detectado: Literal[
        "Pinamar Norte", "Pinamar Centro", "Pinamar General", "Golf", 
        "Alamos/Pioneros", "Ostende", "Valeria del Mar", "Cariló", 
        "La Herradura", "Costa Esmeralda"
    ] = Field(..., description="Selecciona el barrio")
    
    estilo_antiguedad: Literal[
        "Moderna/Nueva", "A estrenar", "Clásica/Tradicional", "Antigua"
    ] = Field(..., description="Estilo o rango de antigüedad")
    
    estado_desc: Literal[
        "excelente", "muy bueno", "bueno", "regular", "a refaccionar"
    ] = Field(..., description="Estado general de conservación")

    # 3. Binarias (Booleanos que se verán como casillas/checkboxes)
    es_a_refaccionar: bool = Field(False, description="¿Está para refaccionar?")
    es_lujosa_hardware: bool = Field(False, description="¿Tiene terminaciones de lujo?")
    dependencia_servicio: bool = Field(False, description="¿Tiene dependencia de servicio?")
    doble_vidrio_dvh: bool = Field(False, description="¿Tiene aberturas DVH?")
    losa_radiante: bool = Field(False, description="¿Tiene losa radiante?")
    es_cercano_mar: bool = Field(False, description="¿Está cerca del mar?")
    gas_natural: bool = Field(False, description="¿Tiene gas natural?")
    es_ph_loft: bool = Field(False, description="¿Es tipo PH o Loft?")
    tiene_seguridad: bool = Field(False, description="¿Cuenta con seguridad?")
    es_renovada: bool = Field(False, description="¿Fue renovada recientemente?")
    en_complejo: bool = Field(False, description="¿Está dentro de un complejo?")
    es_moderna_estrenar: bool = Field(False, description="¿Es moderna o a estrenar?")

    class Config:
        schema_extra = {
            "example": {
                "m2": 850.0,
                "m2_construidos_final": 250.0,
                "ambientes": 5,
                "dormitorios": 3,
                "baños": 3,
                "cocheras": 2,
                "nivel_pileta": 1,
                "cantidad_plantas": 2,
                "barrio_detectado": "Pinamar Norte",
                "estilo_antiguedad": "Moderna/Nueva",
                "estado_desc": "excelente",
                "es_lujosa_hardware": True,
                "losa_radiante": True,
                "gas_natural": True,
                "es_moderna_estrenar": True
            }
        }

@app.post("/predecir")
def predecir_precio(casa: CasaInput):
    try:
        # 1. Pasamos el input a un diccionario manipulable
        datos_entrada = casa.model_dump() if hasattr(casa, 'model_dump') else casa.dict()
        
        # 2. 🧮 AUTOCÁLCULOS INTELIGENTES
        # Calculamos los ratios de forma segura
        dorm_seguros = datos_entrada['dormitorios'] if datos_entrada['dormitorios'] > 0 else 1
        amb_seguros = datos_entrada['ambientes'] if datos_entrada['ambientes'] > 0 else 1
        
        datos_entrada['ratio_bano_dormitorio'] = datos_entrada['baños'] / dorm_seguros
        datos_entrada['ratio_dormitorio_ambiente'] = datos_entrada['dormitorios'] / amb_seguros
        
        # Calculamos los ambientes extra (Ej: Si tiene 5 ambientes y 3 dormitorios, los extras son 2)
        # Usamos max() para asegurarnos de que nunca dé un número negativo por error del usuario
        datos_entrada['ambientes_extra'] = max(0, datos_entrada['ambientes'] - datos_entrada['dormitorios'])

        # 3. Convertimos los booleanos (True/False) nuevamente a enteros (1/0)
        for clave, valor in datos_entrada.items():
            if isinstance(valor, bool):
                datos_entrada[clave] = int(valor)

        # 4. Creamos el DataFrame
        df_entrada = pd.DataFrame([datos_entrada])
        
        # 5. Alinear columnas y relajar tipos para Scikit-Learn
        columnas_esperadas = preprocessor.feature_names_in_
        df_entrada = df_entrada[columnas_esperadas]
        df_entrada = df_entrada.astype(object)
        
        # 6. Transformar y Predecir
        X_proc = preprocessor.transform(df_entrada)
        pred_log = model_gb_log.predict(X_proc)[0]
        precio_estimado = np.expm1(pred_log)
        
        return {
            "precio_estimado_usd": round(precio_estimado, 0),
            "modelo_utilizado": "Gradient Boosting (Log1p)",
            "parametros_autocalculados": {
                "ambientes_extra": datos_entrada['ambientes_extra'],
                "ratio_bano_dormitorio": round(datos_entrada['ratio_bano_dormitorio'], 2),
                "ratio_dormitorio_ambiente": round(datos_entrada['ratio_dormitorio_ambiente'], 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))