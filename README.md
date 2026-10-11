Clasificación de tumores de mama con Machine Learning

## Descripción

Proyecto de clasificación de tumores de mama como benignos o malignos mediante un modelo de regresión logística desarrollado con Python y Scikit-learn. Se utiliza MLflow para registrar experimentos, parámetros, métricas y artefactos.

## Objetivo

Entrenar y evaluar un modelo de clasificación, comparar sus resultados y registrar el proceso experimental para facilitar su seguimiento y reproducibilidad.

## Tecnologías utilizadas

- Python 3.12
- Pandas: carga y preparación de datos.
- Scikit-learn: entrenamiento y evaluación del modelo.
- Matplotlib: visualización de la matriz de confusión.
- MLflow: seguimiento de experimentos y almacenamiento de artefactos.

## Estructura del proyecto

text
dataset/
├── train.py
├── data.csv
├── requirements.txt
├── README.md


## Requisitos previos

- Python 3.12 instalado.
- Archivo `data.csv` disponible en la carpeta del proyecto.
- Dependencias instaladas desde `requirements.txt`.

## Instalación

1. Clonar o descargar el repositorio.
2. Abrir la carpeta del proyecto en VS Code.
3. Crear y activar un entorno virtual:

   bash
   python -m venv .venv
   

   En Windows PowerShell:

   powershell
   .\.venv\Scripts\Activate.ps1
   

4. Instalar las dependencias:

   bash
   python -m pip install -r requirements.txt
   

## Ejecución

Ejecutar el entrenamiento desde la carpeta del proyecto:

bash
python train.py --data data.csv


El programa prepara los datos, divide las muestras en entrenamiento y prueba, entrena el modelo y calcula las métricas de evaluación.

## Seguimiento con MLflow

Para abrir la interfaz de MLflow utilizando el almacenamiento local del proyecto:

bash
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000


Abrir en el navegador:

http://127.0.0.1:5000

El experimento registrado se llama `Breast Cancer Classification`.

Se registran las siguientes métricas:

- Accuracy
- Precision
- Recall
- F1-score

También se guardan los parámetros del modelo y los artefactos generados, como `metrics.json` y `confusion_matrix.png`, además del modelo cuando el registro finaliza correctamente.

## Resultados

En las ejecuciones realizadas, la mejor obtuvo:

- Accuracy: 98.25%
- Precision: 100%
- Recall: 95.24%
- F1-score: 97.56%

Estos resultados corresponden al conjunto de prueba utilizado y no garantizan el mismo desempeño con datos nuevos.

## Limitaciones

Los resultados dependen de los datos y de la partición de entrenamiento y prueba. Se recomienda realizar validación cruzada y experimentos controlados con diferentes hiperparámetros.

**Nota:** Este proyecto tiene fines académicos y no está diseñado para realizar diagnósticos médicos.

## Autoría

Proyecto académico de Machine Learning y seguimiento de experimentos con MLflow.
