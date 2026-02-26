# ML-pinamar-property-valuation

## Descripción del Proyecto
Este proyecto desarrolla un pipeline completo de Machine Learning para estimar el valor de mercado (en USD) de propiedades en la ciudad costera de Pinamar. El objetivo principal es reducir la incertidumbre en las tasaciones inmobiliarias mediante un modelo basado en datos, superando la subjetividad tradicional del sector.

## 🚀 Características Principales y Decisiones Técnicas
* **Origen de Datos y División Temporal:** Los datos fueron recolectados mediante web scraping de la plataforma inmobiliaria más utilizada del país. Para garantizar una evaluación realista y evitar el sobreajuste, se aplicó una división temporal (time-based split): una extracción inicial conformó los conjuntos de entrenamiento (train) y validación (val), mientras que una segunda extracción posterior, compuesta por las publicaciones más recientes, se reservó exclusivamente para el conjunto de prueba (test).
* **Ingeniería de Características (Feature Engineering) Orientada al Negocio:** Creación de variables clave que aportan valor real a la tasación, como `es_lujosa_hardware`, `ratio_bano_dormitorio` y `es_moderna_estrenar`.
* **Prevención de Data Leakage:** Implementación estricta de `ColumnTransformer` y `Pipeline` de Scikit-Learn para aislar el preprocesamiento entre los conjuntos de entrenamiento, validación y prueba.
* **Manejo de Valores Atípicos:** Aplicación de transformación logarítmica al *target* (precio) para estabilizar los gradientes y evitar que las propiedades de ultra-lujo distorsionen el aprendizaje del modelo.
* **Modelado Avanzado:** Entrenamiento y ajuste de hiperparámetros (mediante Random Search y Keras Tuner) de múltiples algoritmos, incluyendo **Gradient Boosting**, **Random Forest** y una **Red Neuronal Robusta** (optimizada con Batch Normalization y LeakyReLU).

## Estructura del Proyecto
El flujo de trabajo está dividido de manera modular en 5 notebooks para simular un entorno de producción ordenado:

1. `cleaning.ipynb`: Limpieza de datos, tratamiento de nulos y corrección de formatos.
2. `feature_engineering.ipynb`: Creación de nuevas variables sintéticas para enriquecer el dataset.
3. `training.ipynb`: Búsqueda de hiperparámetros, entrenamiento de modelos (Regresión, KNN, Ensembles, Deep Learning) y guardado de pipelines (`.pkl` y `.keras`).
4. `validation.ipynb`: Carga de modelos, reversión de la transformación logarítmica y evaluación comparativa en el set de validación.
5. `test_results.ipynb`: Evaluación final e intocable sobre el conjunto de test para medir el rendimiento real frente a datos nunca antes vistos.

## Resultados Destacados
En lugar de depender exclusivamente de métricas teóricas como el R², el modelo fue evaluado mediante métricas de negocio interpretables:
* **Mejor Modelo:** Gradient Boosting Regressor (con target logarítmico).
* **Error Mediano Típico:** ~17% de desvío frente al precio real del mercado, un margen altamente competitivo dada la subjetividad inherente a los precios inmobiliarios.

## Tecnologías Utilizadas
* **Lenguaje:** Python
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, TensorFlow / Keras, Keras Tuner
* **Serialización:** Joblib
* **Visualización:** Matplotlib, Seaborn
