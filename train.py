"""ISTEA · Inteligencia Artificial y Ciencia de Datos.

Asignatura : Laboratorio de Minería de Datos
Profesor   : Diego Mosquera
Práctica   : Experimentación de algoritmos para clasificar el dataset Iris.

PROPÓSITO DOCENTE
-----------------
Demostrar cómo entrenar *múltiples algoritmos y configuraciones* desde un solo
script, sin copiar archivos de entrenamiento. Los algoritmos y sus parámetros
se declaran en config.yaml y cada configuración crea un run de MLflow.

Alcance deliberado: solo experimentación, validación cruzada y MLflow Tracking.
No realiza despliegue, optimización automática ni evaluación final en test.

Uso:
    python train.py
    python train.py --listar
    python train.py --modelo svm_rbf
    python train.py --modelo random_forest_100 --modelo random_forest_300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import sklearn
import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Todas las rutas se resuelven respecto del propio proyecto; no dependen de
# desde qué directorio se ejecute el comando python train.py.
RAIZ = Path(__file__).resolve().parent

# 'Fábrica' de algoritmos: la lógica de train.py NO depende de un modelo fijo.
# El segundo elemento indica si es necesario escalar los atributos de Iris.
# IMPORTANTE: StandardScaler va DENTRO del Pipeline para evitar data leakage
# durante la validación cruzada (se ajusta por separado en cada fold).
ALGORITMOS: dict[str, tuple[type, bool]] = {
    "logistic_regression": (LogisticRegression, True),
    "decision_tree": (DecisionTreeClassifier, False),
    "random_forest": (RandomForestClassifier, False),
    "svm": (SVC, True),
    "knn": (KNeighborsClassifier, True),
    "gaussian_nb": (GaussianNB, False),
    "gradient_boosting": (GradientBoostingClassifier, False),
}

# Multiclase (Iris tiene tres especies). El F1 macro promedia por clase sin
# favorecer a las clases más frecuentes; accuracy complementa la lectura.
METRICAS = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}


# ISTEA | SERIALIZACIÓN SEGURA DE MODELOS GENERADOS LOCALMENTE
# ---------------------------------------------------------------------------
# MLflow 3 usa skops al guardar modelos scikit-learn en un tracking local.
# skops comprueba que el modelo pueda recuperarse sin deserializar tipos
# desconocidos. Algunos algoritmos utilizan estructuras internas de sklearn
# que requieren autorización explícita incluso si el propio script los creó.
#
# Estas listas son EXPLÍCITAS y específicas por algoritmo: no autorizamos
# automáticamente todo lo que reporte skops, ni importamos modelos externos.
# En particular, KNN (con algorithm="auto" en Iris) construye un KDTree y una
# métrica de distancia euclídea. Esto ocasionaba el segundo error reportado.
TIPO_TREE_SKLEARN = "sklearn.tree._tree.Tree"
TIPO_DISTANCIA_KNN = "sklearn.metrics._dist_metrics.EuclideanDistance64"
TIPO_KDTREE_KNN = "sklearn.neighbors._kd_tree.KDTree"

TIPOS_SKOPS_POR_ALGORITMO: dict[str, tuple[str, ...]] = {
    "decision_tree": (TIPO_TREE_SKLEARN,),
    "random_forest": (TIPO_TREE_SKLEARN,),
    "gradient_boosting": (TIPO_TREE_SKLEARN,),
    "knn": (TIPO_DISTANCIA_KNN, TIPO_KDTREE_KNN),
}


def tipos_confiables_skops(algoritmo: str) -> list[str]:
    """Devuelve la lista revisada de tipos internos para el modelo local.

    No acepta tipos suministrados dinámicamente por un archivo de terceros.
    Si se agregan algoritmos o cambian estructuras en una versión de sklearn,
    deben revisarse y probarse explícitamente sus tipos antes de autorizarlos.
    """
    return list(TIPOS_SKOPS_POR_ALGORITMO.get(algoritmo, ()))


def cargar_configuracion(ruta: Path) -> dict[str, Any]:
    """Lee y valida los aspectos fundamentales del contrato config.yaml."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró config.yaml: {ruta}")
    with ruta.open("r", encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)

    if not isinstance(config, dict):
        raise ValueError("config.yaml debe contener un objeto YAML.")
    for seccion in ("institucion", "experimento", "validacion", "modelos"):
        if seccion not in config or not isinstance(config[seccion], dict):
            raise ValueError(f"Falta la sección '{seccion}' en config.yaml.")
    if not config["modelos"]:
        raise ValueError("La sección 'modelos' no puede estar vacía.")

    val = config["validacion"]
    if not 0 < val["proporcion_test"] < 1:
        raise ValueError("'proporcion_test' debe estar estrictamente entre 0 y 1.")
    if not isinstance(val["numero_folds"], int) or val["numero_folds"] < 2:
        raise ValueError("'numero_folds' debe ser un entero >= 2.")
    if val["metrica_principal"] not in METRICAS:
        raise ValueError(f"Métrica principal inválida. Opciones: {list(METRICAS)}")

    # Detectamos nombres de algoritmos no disponibles ANTES de crear runs.
    for nombre, definicion in config["modelos"].items():
        if not isinstance(definicion, dict) or definicion.get("algoritmo") not in ALGORITMOS:
            raise ValueError(f"Algoritmo no soportado en '{nombre}': {definicion}")
        if not isinstance(definicion.get("parametros", {}), dict):
            raise ValueError(f"'parametros' de '{nombre}' debe ser un diccionario.")
    return config


def construir_pipeline(definicion: dict[str, Any]) -> Pipeline:
    """Crea cualquier candidato usando una misma interfaz sklearn Pipeline."""
    nombre_algoritmo = definicion["algoritmo"]
    clase, requiere_escalado = ALGORITMOS[nombre_algoritmo]
    parametros = definicion.get("parametros", {})

    # Cambiar n_estimators, C o el kernel en YAML cambia solo esta instancia.
    # Un nombre de hiperparámetro incorrecto falla pronto con TypeError.
    estimador = clase(**parametros)

    # Para árboles no se requiere StandardScaler; se deja 'passthrough'.
    preprocesamiento = StandardScaler() if requiere_escalado else "passthrough"
    return Pipeline([
        ("escalado", preprocesamiento),
        ("modelo", estimador),
    ])


def preparar_iris(proporcion_test: float, semilla: int):
    """Separa un test intocable; TODO experimento usa solo X_train/y_train."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=proporcion_test,
        stratify=iris.target,  # Conserva la proporción de las tres clases.
        random_state=semilla,  # La separación es reproducible.
    )
    return iris, X_train, X_test, y_train, y_test


def huella_dataset(iris) -> str:
    """Fingerprint SHA-256 de valores/features+targets del Iris cargado."""
    resumen = hashlib.sha256()
    resumen.update(np.ascontiguousarray(iris.data, dtype="<f8").tobytes())
    resumen.update(np.ascontiguousarray(iris.target, dtype="<i4").tobytes())
    return resumen.hexdigest()


def commit_git() -> str:
    """Registra el commit solo si este ZIP se convirtió en un repo Git."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RAIZ, capture_output=True, text=True, check=False, timeout=3,
        )
        return resultado.stdout.strip() if resultado.returncode == 0 else "sin_repositorio"
    except (OSError, subprocess.TimeoutExpired):
        return "git_no_disponible"


def iniciar_tracking(config: dict[str, Any]) -> str:
    """Crea/reutiliza un experimento con SQLite y artefactos 100 % locales."""
    exp_cfg = config["experimento"]
    db = (RAIZ / exp_cfg["base_de_datos"]).resolve()
    artefactos = (RAIZ / exp_cfg["directorio_artefactos"]).resolve()
    artefactos.mkdir(parents=True, exist_ok=True)

    # SQLite guarda métricas, params, tags y runs (sin servidor obligatorio).
    # Ejemplo Linux: sqlite:////ruta/proyecto/mlflow.db
    # Ejemplo Windows: sqlite:///C:/ruta/proyecto/mlflow.db
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")

    nombre = exp_cfg["nombre"]
    existente = mlflow.get_experiment_by_name(nombre)
    if existente is None:
        # CLAVE: fijamos la carpeta de artefactos explícitamente. Así el
        # script puede registrar modelos sin depender de un servidor remoto.
        exp_id = mlflow.create_experiment(
            name=nombre,
            artifact_location=artefactos.as_uri(),
            tags={
                "institucion": config["institucion"]["nombre"],
                "asignatura": config["institucion"]["asignatura"],
                "profesor": config["institucion"]["profesor"],
                "dataset": "Iris (scikit-learn)",
            },
        )
    else:
        exp_id = existente.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def evaluar_validacion_cruzada(pipeline: Pipeline, X_train, y_train,
                               numero_folds: int, semilla: int) -> tuple[dict, dict]:
    """Devuelve métricas agregadas y los valores de cada fold (sin usar test)."""
    cv = StratifiedKFold(n_splits=numero_folds, shuffle=True, random_state=semilla)

    # cross_validate CLONA el pipeline en cada fold: el scaler se ajusta
    # solo con la partición de entrenamiento de ese fold, nunca con su validación.
    resultados_cv = cross_validate(
        pipeline, X_train, y_train,
        scoring=METRICAS,
        cv=cv,
        n_jobs=1,           # Más simple/reproducible para una práctica docente.
        error_score="raise",  # Evita ocultar errores de modelos/configuración.
        return_train_score=False,
    )

    medias_y_dispersion = {}
    metricas_folds = {}
    for nombre_metrica in METRICAS:
        valores = resultados_cv[f"test_{nombre_metrica}"]
        medias_y_dispersion[f"cv_{nombre_metrica}_mean"] = float(np.mean(valores))
        medias_y_dispersion[f"cv_{nombre_metrica}_std"] = float(np.std(valores))
        metricas_folds[nombre_metrica] = [float(v) for v in valores]
    return medias_y_dispersion, metricas_folds


def ejecutar_configuracion(nombre: str, definicion: dict, config: dict,
                            iris, X_train, y_train, fingerprint: str) -> dict:
    """Una configuración de YAML = una ejecución (run) de MLflow."""
    val = config["validacion"]
    pipeline = construir_pipeline(definicion)

    # MLflow inicia/termina automáticamente el run con este bloque 'with'.
    # Si ocurre un error, MLflow marca la ejecución como fallida.
    with mlflow.start_run(run_name=nombre) as run:
        # Tags describen contexto; params describen configuración entrenable.
        # Registrar versiones y el hash evita runs difíciles de reproducir.
        mlflow.set_tags({
            "institucion": config["institucion"]["nombre"],
            "carrera": config["institucion"]["carrera"],
            "asignatura": config["institucion"]["asignatura"],
            "profesor": config["institucion"]["profesor"],
            "dataset": "Iris",
            "dataset_sha256": fingerprint,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "mlflow_version": mlflow.__version__,
            "git_commit": commit_git(),
        })

        # Los hiperparámetros EFECTIVOS incluyen los defaults del estimador,
        # no solamente los parámetros sobrescritos explícitamente en YAML.
        hiperparametros = {
            f"hp_{clave}": str(valor)
            for clave, valor in pipeline.named_steps["modelo"].get_params(deep=False).items()
        }
        mlflow.log_params({
            "algoritmo": definicion["algoritmo"],
            "configuracion": nombre,
            "semilla_split_cv": val["semilla"],
            "numero_folds": val["numero_folds"],
            "proporcion_test_reservado": val["proporcion_test"],
            "escalado": "StandardScaler" if pipeline.named_steps["escalado"] != "passthrough" else "no",
            **hiperparametros,
        })

        # La configuración original queda como artefacto legible por alumnos.
        mlflow.log_dict({
            "institucion": config["institucion"],
            "dataset": "load_iris() de scikit-learn",
            "configuracion": nombre,
            "modelo": definicion,
            "validacion": val,
            "features": list(iris.feature_names),
            "clases": list(iris.target_names),
        }, "configuracion_ejecucion.json")

        # Comparación JUSTA: todos los candidatos se validan con los mismos
        # folds, que se crean con la misma semilla sobre el mismo X_train.
        metricas, por_fold = evaluar_validacion_cruzada(
            pipeline, X_train, y_train, val["numero_folds"], val["semilla"]
        )
        mlflow.log_metrics(metricas)
        mlflow.log_dict(por_fold, "metricas_por_fold.json")

        # Reentrenamos sobre TODO X_train únicamente para conservar el
        # artefacto del candidato. X_test permanece completamente sin usar.
        pipeline.fit(X_train, y_train)
        # MLflow 3 usa skops para serializar en tracking local. Skops verifica
        # mediante dump/load que los tipos internos del modelo sean conocidos.
        # Tipos internos autorizados por FAMILIA de algoritmo, no una lista
        # global sin restricciones: Tree para árboles, KDTree y distancia
        # euclídea para KNN en este dataset. NO cargamos artefactos externos.
        modelo_info = mlflow.sklearn.log_model(
            pipeline,
            name="modelo_iris",
            input_example=X_train[:3],
            serialization_format="skops",
            skops_trusted_types=tipos_confiables_skops(definicion["algoritmo"]),
        )

        # En MLflow 3, la referencia canónica para recuperar el modelo es
        # modelo_info.model_uri, no una ruta inventada dentro del run.
        salida = {
            "configuracion": nombre,
            "algoritmo": definicion["algoritmo"],
            "run_id": run.info.run_id,
            "model_uri": modelo_info.model_uri,
            **metricas,
        }
        print(
            f"[OK] {nombre:<24} | F1 macro CV = "
            f"{metricas['cv_f1_macro_mean']:.4f} "
            f"(+/- {metricas['cv_f1_macro_std']:.4f}) | run_id={run.info.run_id}"
        )
        return salida


def guardar_resumen(resultados: list[dict]) -> Path:
    """CSV local adicional para lectura rápida; MLflow es la fuente histórica."""
    directorio = RAIZ / "reports"
    directorio.mkdir(exist_ok=True)
    destino = directorio / "ultima_ejecucion.csv"
    with destino.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0].keys()))
        escritor.writeheader()
        escritor.writerows(resultados)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISTEA · Laboratorio de Minería de Datos · Experimentación Iris + MLflow"
    )
    parser.add_argument("--listar", action="store_true",
                        help="Muestra las configuraciones disponibles sin entrenar.")
    parser.add_argument("--modelo", action="append", metavar="NOMBRE",
                        help="Ejecuta una configuración; repetir para seleccionar varias.")
    args = parser.parse_args()

    config = cargar_configuracion(RAIZ / "config.yaml")
    modelos = config["modelos"]
    if args.listar:
        print("Configuraciones disponibles (config.yaml):")
        for nombre, definicion in modelos.items():
            print(f"  {nombre:<24} -> {definicion['algoritmo']}")
        return

    elegidos = args.modelo if args.modelo else list(modelos)
    desconocidos = sorted(set(elegidos) - set(modelos))
    if desconocidos:
        parser.error(f"Configuraciones inexistentes: {desconocidos}. Usa --listar.")
    # Evitamos ejecutar accidentalmente dos veces el mismo nombre en un comando.
    elegidos = list(dict.fromkeys(elegidos))

    val = config["validacion"]
    iris, X_train, X_test, y_train, y_test = preparar_iris(
        val["proporcion_test"], val["semilla"]
    )
    fingerprint = huella_dataset(iris)
    experimento_id = iniciar_tracking(config)

    print("=" * 78)
    print("ISTEA | Inteligencia Artificial y Ciencia de Datos")
    print("Laboratorio de Minería de Datos | Profesor: Diego Mosquera")
    print(f"Iris: {len(iris.data)} registros, {len(iris.feature_names)} features, "
          f"{len(iris.target_names)} clases")
    print(f"Train: {len(X_train)} | Test reservado: {len(X_test)} "
          f"(no se usa para seleccionar modelos)")
    print(f"Experimento MLflow: {config['experimento']['nombre']} (id={experimento_id})")
    print("=" * 78)

    resultados = []
    for nombre in elegidos:
        salida = ejecutar_configuracion(
            nombre, modelos[nombre], config, iris, X_train, y_train, fingerprint
        )
        resultados.append(salida)

    # La selección es una ORIENTACIÓN EXPERIMENTAL, no una evaluación en test.
    # El criterio de desempate prioriza menor dispersión (más estabilidad).
    metrica = val["metrica_principal"]
    ordenados = sorted(
        resultados,
        key=lambda r: (
            -r[f"cv_{metrica}_mean"],
            r[f"cv_{metrica}_std"],
            -r["cv_accuracy_mean"],
            r["configuracion"],
        ),
    )
    reporte = guardar_resumen(ordenados)
    ganador = ordenados[0]
    print("\nRESUMEN DE LA EJECUCIÓN ACTUAL (NO de todos los runs históricos)")
    for posicion, r in enumerate(ordenados, start=1):
        print(f"  {posicion:>2}. {r['configuracion']:<24} "
              f"F1 macro={r['cv_f1_macro_mean']:.4f} "
              f"std={r['cv_f1_macro_std']:.4f}")
    print(f"\nCandidato mejor clasificado por CV: {ganador['configuracion']}")
    print(f"CSV local: {reporte}")
    print("La evaluación final en el test reservado NO forma parte de esta práctica.")
    print("Ver resultados: mlflow server --backend-store-uri sqlite:///mlflow.db "
          "--host 127.0.0.1 --port 5000")
    print("Abrir: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
