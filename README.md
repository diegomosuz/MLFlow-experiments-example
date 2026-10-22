# ISTEA · Laboratorio de Minería de Datos

## Práctica: Experimentación de algoritmos de clasificación con Iris y MLflow

**Institución:** ISTEA  
**Carrera:** Inteligencia Artificial y Ciencia de Datos  
**Asignatura:** Laboratorio de Minería de Datos  
**Profesor:** Diego Mosquera  
**Modalidad:** práctica local y reproducible, orientada a la etapa **experimental** del ciclo de vida de Machine Learning.

---

## 1. Objetivo de la práctica

Aprender a comparar múltiples algoritmos de *Machine Learning tradicional* **sin crear un script de entrenamiento por algoritmo**. Todo el trabajo utiliza:

- **`train.py`**: un solo motor genérico que prepara Iris, construye el estimador, ejecuta validación cruzada y registra resultados.
- **`config.yaml`**: contiene los modelos candidatos y sus hiperparámetros. Es el principal punto de modificación del estudiante.
- **MLflow Tracking**: registra y conserva cada ejecución (*run*) con sus parámetros, métricas, etiquetas y modelo entrenado.
- **scikit-learn**: aporta el dataset Iris, los estimadores, los pipelines y la validación cruzada.

**Alcance:** experimentación y comparación de modelos. No hay despliegue, serving, CI/CD, registro de modelos en producción ni reentrenamiento automático.

### El problema: clasificación Iris

Iris contiene **150 muestras**, **4 atributos numéricos** (medidas de sépalos y pétalos) y **3 clases** de flores (setosa, versicolor y virginica). Se busca predecir la especie a partir de las medidas.

La función `sklearn.datasets.load_iris()` suministra los datos localmente. **No es necesario descargar ningún CSV ni configurar credenciales, servidores externos o APIs.**

---

## 2. ¿Qué se va a experimentar?

El `config.yaml` inicial contiene **9 configuraciones** correspondientes a **7 familias de algoritmos**:

| Configuración en YAML | Algoritmo | Particularidad |
| --- | --- | --- |
| `regresion_logistica` | Logistic Regression | Escalado de variables |
| `arbol_decision` | Decision Tree | Profundidad máxima de 4 |
| `random_forest_100` | Random Forest | 100 árboles |
| `random_forest_300` | Random Forest | 300 árboles |
| `svm_lineal` | Support Vector Machine | Kernel lineal, escalado |
| `svm_rbf` | Support Vector Machine | Kernel RBF, escalado |
| `knn_5_vecinos` | K-Nearest Neighbors | 5 vecinos, escalado |
| `naive_bayes_gaussiano` | Gaussian Naive Bayes | Varianza suavizada |
| `gradient_boosting` | Gradient Boosting | 100 estimadores |

**Regla principal:** una configuración = un **run** independiente en MLflow. Se repite la ejecución de `train.py` para agregar nuevos runs, nunca para sobrescribir runs previos.

---

## 3. Archivos incluidos

```text
ISTEA_Laboratorio_Mineria_Datos_Iris_MLflow/
│
├── README.md              # Guía completa para el estudiante
├── train.py               # ÚNICO script de experimentación
├── config.yaml            # Modelos, hiperparámetros y validación
├── requirements.txt       # Dependencias de Python
├── .gitignore             # Exclusiones para Git
└── tests/
    └── test_train.py      # Pruebas automáticas básicas
```

Tras ejecutar la práctica, aparecerán automáticamente:

```text
├── mlflow.db              # Base SQLite: runs, parámetros, métricas, tags
├── mlartifacts/            # Artefactos: archivos JSON y modelos MLflow
└── reports/
    └── ultima_ejecucion.csv  # Resumen de la ÚLTIMA invocación de train.py
```

**Importante:** `mlflow.db` y `mlartifacts/` son carpetas/archivos de ejecución. **No vienen precargados en el ZIP**, porque la actividad consiste justamente en generarlos. Están ignorados en `.gitignore`.

---

## 4. Preparación del entorno Python

### Requisitos previos

1. Computadora con **Windows 10/11, macOS o Linux**.
2. **Python 3.11 o 3.12 (recomendado)** instalado y accesible por consola.
3. Conexión a Internet **solo para instalar los paquetes con pip**, si aún no están disponibles.
4. Una terminal y un navegador.

> **¿Por qué un entorno virtual?** Para que las versiones de MLflow, scikit-learn y demás paquetes de esta práctica no modifiquen otros proyectos Python instalados en la computadora.

### 4.1. Windows: PowerShell

**Paso A.** Descomprimir el ZIP en una carpeta, por ejemplo `C:\Cursos\ISTEA\`. Abrir PowerShell y entrar a la carpeta del proyecto:

```powershell
cd "C:\Cursos\ISTEA\ISTEA_Laboratorio_Mineria_Datos_Iris_MLflow"
```

> La ruta es un **ejemplo**. Usar la ruta real donde se descomprimió el ZIP.

**Paso B.** Verificar las versiones disponibles:

```powershell
py -0p
python --version
```

**Paso C.** Crear el entorno virtual. Si se dispone de Python 3.11:

```powershell
py -3.11 -m venv .venv
```

O, para Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Si el comando `py` no está disponible pero `python` apunta a una versión compatible:

```powershell
python -m venv .venv
```

**Paso D.** Activar el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

La consola debería mostrar el prefijo `(.venv)`.

Si PowerShell impide ejecutar `Activate.ps1`, habilitar **solamente para esta sesión**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Paso E.** Instalar dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Paso F.** Comprobar el entorno:

```powershell
python --version
python -m pip show mlflow scikit-learn PyYAML
```

**Al finalizar la sesión**, opcionalmente desactivar el entorno con `deactivate`.

### 4.2. Linux o macOS: terminal Bash/Zsh

Descomprimir, entrar en el directorio real del proyecto y ejecutar:

```bash
cd /ruta/al/ISTEA_Laboratorio_Mineria_Datos_Iris_MLflow
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si hay varias versiones de Python, usar explícitamente `python3.11` o `python3.12` para crear el entorno.

> Todas las órdenes siguientes se ejecutan con **el entorno activado** y dentro de la **carpeta raíz del proyecto**, salvo indicación contraria.

---

## 5. Ejecutar el experimento paso a paso

### Paso 1. Ver las configuraciones (sin entrenar)

```bash
python train.py --listar
```

Verás las nueve configuraciones declaradas en `config.yaml`. **Este comando no crea runs ni entrena modelos**.

### Paso 2. Entrenar y registrar todas las configuraciones

```bash
python train.py
```

Esto realiza, en orden:

1. Lee y valida `config.yaml`.
2. Carga Iris desde scikit-learn.
3. Separa los datos **una sola vez**: 80 % train y 20 % test reservado, estratificados.
4. Configura el tracking local de MLflow sobre SQLite.
5. Recorre las 9 configuraciones.
6. Por cada configuración, construye un `Pipeline` y ejecuta **validación cruzada estratificada con 5 folds** utilizando **solo train**.
7. Registra en un run de MLflow el algoritmo, los hiperparámetros, las métricas, los valores por fold, la configuración y el modelo entrenado con todo train.
8. Ordena los resultados de esa ejecución por F1 macro promedio, con desempates por dispersión y accuracy, y genera un CSV.

**Salida esperada:** verás líneas similares a:

```text
[OK] regresion_logistica      | F1 macro CV = 0.xxxx (+/- 0.xxxx) | run_id=...
[OK] arbol_decision           | F1 macro CV = 0.xxxx (+/- 0.xxxx) | run_id=...
...
Candidato mejor clasificado por CV: <nombre_de_la_configuracion>
CSV local: .../reports/ultima_ejecucion.csv
```

Los valores mostrados son **ilustrativos**; los valores reales se calculan al ejecutar el código. No se garantiza de antemano cuál será el algoritmo ganador.

### Paso 3. Ejecutar una sola configuración (opcional)

Si deseas probar una alternativa sin volver a entrenar todas:

```bash
python train.py --modelo svm_rbf
```

O dos configuraciones concretas:

```bash
python train.py --modelo random_forest_100 --modelo random_forest_300
```

Cada nueva invocación agrega runs a MLflow. Si repites el mismo nombre, **MLflow mantiene dos run IDs distintos** con el mismo nombre visible.

### Paso 4. Ejecutar los tests automáticos (opcional)

```bash
python -m unittest discover -s tests -v
```

Comprueban las siete familias de algoritmos, el escalado, la partición de Iris,
la validación cruzada y la lista restringida de tipos autorizados por `skops`.
También incluyen una **prueba de serialización y recuperación real** de los
modelos basados en árboles, que reproduce el paso de seguridad ejecutado por
MLflow. Si `skops` no está instalado, esa prueba se omite explícitamente.
Estas pruebas no necesitan arrancar la interfaz web de MLflow.

---

## 6. Consultar los experimentos en MLflow UI

**Paso A.** Mantén los experimentos ya generados en disco. Abre **una segunda terminal** y sitúate también en la carpeta raíz del proyecto. Activa en ella el mismo entorno virtual.

Windows PowerShell:

```powershell
cd "C:\Cursos\ISTEA\ISTEA_Laboratorio_Mineria_Datos_Iris_MLflow"
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
cd /ruta/al/ISTEA_Laboratorio_Mineria_Datos_Iris_MLflow
source .venv/bin/activate
```

**Paso B.** Levanta el servidor y la interfaz web de MLflow apuntando a la **misma base SQLite**:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

**Paso C.** Abre el navegador en:

**http://127.0.0.1:5000**

**Paso D.** Dentro de MLflow:

1. Abre el experimento `ISTEA_Laboratorio_Mineria_de_Datos_Iris`.
2. Observa sus **runs** (uno por configuración entrenada).
3. Personaliza la tabla para incluir las métricas `cv_f1_macro_mean`, `cv_f1_macro_std` y `cv_accuracy_mean`.
4. Ordena por **`cv_f1_macro_mean` descendente** para comparar.
5. Entra en un run y revisa sus **Parameters**, **Metrics** y **Tags**.
6. Examina `configuracion_ejecucion.json` y `metricas_por_fold.json` entre los artefactos de cada run.
7. Examina el **modelo registrado como logged model**, vinculado a ese run. En MLflow 3 los modelos guardados pueden consultarse desde la sección de modelos y mediante la `model_uri` asociada al run; **no tiene por qué aparecer una carpeta tradicional `run/artifacts/model`**.
8. Selecciona varios runs en la tabla y utiliza las opciones de **comparación** de MLflow para contrastar métricas y parámetros.

> No hace falta iniciar el servidor antes de `python train.py`: el script escribe directamente en SQLite. El servidor se utiliza aquí como **interfaz de consulta** sobre los mismos datos locales.

**Detener el servidor:** volver a la segunda terminal y pulsar `Ctrl+C`.

### ¿Dónde guarda cada cosa MLflow?

| Componente | Ruta local | Qué conserva |
| --- | --- | --- |
| **Backend store** | `mlflow.db` | Experimentos, runs, parámetros, métricas y etiquetas |
| **Artifact store** | `mlartifacts/` | JSON con configuración y folds, modelos serializados |
| **Resumen adicional** | `reports/ultima_ejecucion.csv` | Solo las configuraciones del último comando |

El CSV **no** es el historial de experimentos. La fuente de consulta histórica es MLflow.

---

## 7. Modificar la experimentación SIN modificar `train.py`

Abrir `config.yaml`, localizar `modelos:` y agregar una nueva entrada. Por ejemplo, un KNN con nueve vecinos:

```yaml
  knn_9_vecinos:
    algoritmo: knn
    parametros:
      n_neighbors: 9
      weights: distance
```

Debe mantenerse la **misma indentación** que las otras entradas dentro de `modelos:` (dos espacios antes de `knn_9_vecinos`).

Después ejecutar:

```bash
python train.py --modelo knn_9_vecinos
```

MLflow añadirá un run independiente, cuyo nombre será `knn_9_vecinos`. El script Python permanece intacto.

Para comparar regularizaciones diferentes, puedes añadir por ejemplo:

```yaml
  regresion_logistica_c_01:
    algoritmo: logistic_regression
    parametros:
      C: 0.1
      max_iter: 2000
      random_state: 42
```

**Consejo:** da a cada variante un nombre descriptivo. Si modificas los parámetros de una entrada existente y la ejecutas de nuevo, también se registra un nuevo run; MLflow conserva los parámetros efectivos, pero el nombre de la variante podría resultar ambiguo.

**Agregar un algoritmo no incluido en la fábrica:** sí requiere incorporar una nueva clase a `ALGORITMOS` en `train.py`, porque `config.yaml` solo parametriza los algoritmos soportados. Para agregar nuevas **configuraciones** de los siete algoritmos incluidos, no hay que tocar código Python.

---

## 8. Entender las buenas prácticas aplicadas

### 8.1. Separamos experimentación de algoritmos

El bucle de `train.py` **no contiene siete implementaciones distintas de entrenamiento**. Construye cualquier modelo declarado con un contrato común (`fit`/`predict`) y utiliza `cross_validate` de scikit-learn. Esto reduce duplicación e inconsistencias.

### 8.2. Un único criterio comparable

Todos utilizan la **misma partición train** y, gracias a `StratifiedKFold(shuffle=True, random_state=42)`, los mismos cinco folds. Se comparan por:

- **`cv_f1_macro_mean`**: métrica principal en este ejemplo multiclase.
- **`cv_f1_macro_std`**: dispersión de los resultados entre folds.
- **`cv_accuracy_mean`**: métrica complementaria.
- También se registran precision y recall macro, y sus desviaciones.

**Nota:** la desviación entre folds es una señal descriptiva de estabilidad, no una garantía estadística de generalización.

### 8.3. Evitamos data leakage

Los modelos que necesitan escalado (`regresion_logistica`, `svm`, `knn`) lo incorporan como etapa de `Pipeline`. El escalador se ajusta **dentro de cada fold de entrenamiento**, sin acceso a la parte de validación. No se escala el dataset completo antes de hacer los splits.

### 8.4. El test queda reservado

Se conserva un **20 % de Iris como holdout/test**, pero **no se calcula ninguna métrica de test en este proyecto**. La elección provisional del candidato se hace con validación cruzada del **80 % de train**.

En un proyecto real, tras congelar la selección y el procedimiento se haría **una evaluación final** con ese test reservado. Reutilizarlo durante muchas rondas de selección introduce sesgo. Deliberadamente no implementamos esa etapa porque esta práctica trata exclusivamente sobre experimentación y tracking.

### 8.5. Registro explícito y trazabilidad

Para cada run se guardan:

| Tipo | Ejemplos | Para qué sirve |
| --- | --- | --- |
| **Parameters** | `algoritmo`, `hp_n_estimators`, `hp_C`, `numero_folds` | Repetir la configuración |
| **Metrics** | `cv_f1_macro_mean`, `cv_f1_macro_std`, `cv_accuracy_mean` | Comparar candidatos |
| **Tags** | Institución, asignatura, profesor, versiones, commit Git, huella del dataset | Documentar contexto y origen |
| **Artifacts** | JSON de configuración, métricas por fold, pipeline/modelo MLflow | Auditar y recuperar resultados |

El código utiliza **logging explícito** (`mlflow.log_params`, `mlflow.log_metrics`, `mlflow.log_dict` y `mlflow.sklearn.log_model`) para que el estudiante entienda exactamente qué se guarda. No se requiere `mlflow.sklearn.autolog()` en este ejercicio.

### 8.6. ¿Por qué aquí no usamos DVC?

Iris viene incorporado en scikit-learn y no tiene un archivo de datos externo que deba evolucionar por versiones. En este caso, añadir DVC sería complejidad innecesaria. El script registra una **huella SHA-256** calculada sobre los datos y las etiquetas, además de las versiones de las dependencias.

Para datasets reales que cambian con frecuencia, es razonable integrar DVC y registrar su revisión o hash en MLflow. Esa ampliación queda fuera de esta práctica.

### 8.7. Persistencia: ¿qué pasa al cerrar la PC?

`mlflow.db` y `mlartifacts/` permanecen en el disco, aun cuando se cierre la terminal. Al volver a abrir MLflow UI desde el mismo proyecto y contra esa misma base SQLite, reaparecen los experimentos previos.

---

## 9. Propuesta de actividad para los estudiantes

1. Ejecutar las nueve configuraciones y ubicar los nueve runs en MLflow.
2. Identificar la mejor configuración por `cv_f1_macro_mean`.
3. Comparar `random_forest_100` y `random_forest_300`: ¿más árboles mejoraron el F1 de validación?
4. Comparar `svm_lineal` y `svm_rbf`: ¿cambiar el kernel modificó el resultado?
5. Crear `knn_9_vecinos` sin editar `train.py` y registrar su nuevo run.
6. Abrir las métricas de cada fold y explicar por qué no basta mirar una sola corrida.
7. Explicar qué información aporta cada uno de los conceptos **Experiment, Run, Parameters, Metrics, Tags y Artifacts**.

**Entrega sugerida:** una captura de la tabla comparativa de MLflow, una conclusión corta sobre la configuración mejor evaluada y la justificación de que el test no se utilizó para seleccionar el ganador.

---

## 10. Problemas habituales y soluciones

| Problema | Qué verificar |
| --- | --- |
| `No module named mlflow` | Activar `.venv` e instalar `requirements.txt` con `python -m pip` del entorno correcto. |
| `Activate.ps1 cannot be loaded` | En PowerShell aplicar `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y reintentar. |
| `No se encontró config.yaml` | Confirmar que no se movió/eliminó el archivo y que se descomprimió todo el ZIP. |
| `Algoritmo no soportado` | Utilizar uno de los identificadores admitidos en `ALGORITMOS`, no el nombre informal de una clase. |
| `Unexpected keyword argument` | El nombre de algún hiperparámetro en YAML no corresponde al estimador de scikit-learn. |
| No aparece el experimento en la UI | Iniciar MLflow desde la **raíz del proyecto** y con `--backend-store-uri sqlite:///mlflow.db` (misma BD usada por el script). |
| El puerto 5000 está ocupado | Usar `--port 5001` y navegar a `http://127.0.0.1:5001`. |
| Aparecen varios runs con el mismo nombre | Es normal: cada ejecución crea un run ID nuevo, incluso cuando se repite la configuración. |
| La base existe, pero los artefactos faltan tras mover carpetas | Las ubicaciones de artefactos se establecen cuando se crea el experimento. Conviene mantener junta la carpeta del proyecto y no moverla sin revisar esas referencias. |
| `UntrustedTypesFoundException` al guardar un árbol o KNN | La versión actual de `train.py` incorpora listas explícitas de tipos confiables para ambos casos, restringidas a los modelos entrenados por este proyecto. Ejecuta la prueba integral `python -m unittest discover -s tests -v` antes de repetir el entrenamiento. |

### Por qué MLflow/skops rechazaba árboles y KNN

**La primera corrección solucionó los árboles, pero no incluyó KNN.**
`skops` (utilizado por MLflow 3 en este entorno) comprueba que se pueda
recuperar el modelo persistido. Determinados algoritmos requieren clases
internas de scikit-learn que no forman parte de su lista de tipos confiables.

Este proyecto solo autoriza los tipos necesarios para las configuraciones
incluidas y **entrenadas localmente**:

| Algoritmos | Tipos internos autorizados |
| --- | --- |
| Decision Tree, Random Forest, Gradient Boosting | `sklearn.tree._tree.Tree` |
| KNN (Iris, búsqueda automática KDTree) | `sklearn.metrics._dist_metrics.EuclideanDistance64` y `sklearn.neighbors._kd_tree.KDTree` |
| Regresión logística, SVM y GaussianNB | Ninguno adicional |

La operación importante es:

```python
mlflow.sklearn.log_model(
    pipeline,
    name="modelo_iris",
    input_example=X_train[:3],
    serialization_format="skops",
    skops_trusted_types=tipos_confiables_skops(definicion["algoritmo"]),
)
```

La prueba `test_skops_roundtrip_de_todos_los_candidatos` serializa y
recupera **las nueve configuraciones**, no solo la que había fallado, para
detectar previamente incompatibilidades de tipos internos. Si `skops` no está
instalado, la prueba informa `skipped`; para validar el proyecto completamente
debe ejecutarse en un entorno que tenga instaladas las dependencias.

**Seguridad:** estas autorizaciones son adecuadas exclusivamente para modelos
propios de procedencia conocida; no cargar modelos externos sin verificar sus
tipos y origen. Tampoco es recomendable autorizar automáticamente todo lo que
reporte `skops.get_untrusted_types()`: cualquier tipo nuevo debe revisarse.

Si ejecutaste una versión anterior, **no borres** `mlflow.db` ni `mlartifacts/`;
las ejecuciones `FINISHED` y `FAILED` anteriores se conservan. Puedes
reemplazar `train.py` y la carpeta `tests/` por los de este ZIP y ejecutar:

```powershell
# 1. Confirma la versión instalada y prueba la serialización de TODOS los modelos.
python -m unittest discover -s tests -v

# 2. Reintenta solo el algoritmo que estaba fallando.
python train.py --modelo knn_5_vecinos

# 3. Cuando la prueba individual funcione, vuelve a ejecutar los nueve.
python train.py
```

Los nuevos runs tienen nuevos IDs; repetir las pruebas agrega runs al histórico
sin sobrescribir los anteriores. El primer comando debe indicar la ejecución
correcta de la prueba de roundtrip de `skops`, no `skipped`.

### Notas sobre versiones

El proyecto utiliza **MLflow 3.x**, en particular la API moderna `mlflow.sklearn.log_model(..., name="modelo_iris")`. Algunas guías antiguas de MLflow 2 muestran otra organización de los archivos del modelo. La documentación oficial vigente explica esas diferencias.

Documentación de referencia:

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow: seguimiento en SQLite local](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database)
- [MLflow: registro de modelos scikit-learn](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
- [scikit-learn: cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
- [scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

---

**ISTEA · Carrera de Inteligencia Artificial y Ciencia de Datos**  
**Laboratorio de Minería de Datos · Profesor Diego Mosquera**
