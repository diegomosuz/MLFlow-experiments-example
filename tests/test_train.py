"""Pruebas docentes | ISTEA | Laboratorio de Minería de Datos.

Profesor: Diego Mosquera.
Objetivo: verificar las decisiones fundamentales del motor experimental
sin arrancar la interfaz web de MLflow ni descargar datasets externos.
"""

import unittest
import tempfile
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import (ALGORITMOS, RAIZ, TIPO_TREE_SKLEARN,
                   TIPO_DISTANCIA_KNN, TIPO_KDTREE_KNN,
                   cargar_configuracion, construir_pipeline,
                   evaluar_validacion_cruzada, preparar_iris,
                   huella_dataset, tipos_confiables_skops)


class PruebasExperimentacionIris(unittest.TestCase):
    """Comprueba contrato YAML, preprocesamiento y validación cruzada."""

    @classmethod
    def setUpClass(cls):
        cls.config = cargar_configuracion(RAIZ / "config.yaml")

    def test_hay_al_menos_siete_familias_y_nueve_configuraciones(self):
        self.assertGreaterEqual(len(ALGORITMOS), 7)
        self.assertGreaterEqual(len(self.config["modelos"]), 9)

    def test_modelos_que_requieren_escalado_lo_reciben(self):
        for nombre in ("regresion_logistica", "svm_rbf", "knn_5_vecinos"):
            with self.subTest(configuracion=nombre):
                pipe = construir_pipeline(self.config["modelos"][nombre])
                self.assertIsInstance(pipe, Pipeline)
                self.assertIsInstance(pipe.named_steps["escalado"], StandardScaler)

    def test_random_forest_no_escala_y_respeta_hiperparametros(self):
        pipe = construir_pipeline(self.config["modelos"]["random_forest_300"])
        self.assertEqual(pipe.named_steps["escalado"], "passthrough")
        self.assertEqual(pipe.named_steps["modelo"].n_estimators, 300)

    def test_reserva_holdout_y_fingerprint_reproducible(self):
        iris, X_train, X_test, y_train, y_test = preparar_iris(0.20, 42)
        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(len(y_train) + len(y_test), 150)
        self.assertEqual(huella_dataset(iris), huella_dataset(iris))

    def test_cv_devuelve_metricas_y_cinco_folds(self):
        _iris, X_train, _X_test, y_train, _y_test = preparar_iris(0.20, 42)
        pipe = construir_pipeline(self.config["modelos"]["regresion_logistica"])
        metrics, per_fold = evaluar_validacion_cruzada(pipe, X_train, y_train, 5, 42)
        self.assertIn("cv_f1_macro_mean", metrics)
        self.assertIn("cv_f1_macro_std", metrics)
        self.assertEqual(len(per_fold["f1_macro"]), 5)
        self.assertTrue(0 <= metrics["cv_f1_macro_mean"] <= 1)


    def test_skops_tipos_explicitos_para_cada_familia(self):
        """Previene errores de serialización por tipos internos de sklearn."""
        for nombre in ("decision_tree", "random_forest", "gradient_boosting"):
            with self.subTest(algoritmo=nombre):
                self.assertEqual(tipos_confiables_skops(nombre), [TIPO_TREE_SKLEARN])

        # KNN sobre Iris usa KDTree y métrica euclídea: ambos tipos deben
        # autorizarse para que MLflow/skops pueda persistir y leer el modelo.
        self.assertEqual(
            tipos_confiables_skops("knn"),
            [TIPO_DISTANCIA_KNN, TIPO_KDTREE_KNN],
        )
        for nombre in ("logistic_regression", "svm", "gaussian_nb"):
            with self.subTest(algoritmo=nombre):
                self.assertEqual(tipos_confiables_skops(nombre), [])

    def test_skops_roundtrip_de_todos_los_candidatos(self):
        """Comprueba serialización Y deserialización real de los nueve modelos.

        Es el test de REGRESIÓN principal: un algoritmo que entrene y prediga
        correctamente puede fallar al registrarse en MLflow. Por eso
        verificamos cada Pipeline con skops.dump + skops.load, equivalentes
        a la comprobación de confianza que realiza MLflow internamente.
        """
        try:
            import skops.io
        except ImportError:
            self.skipTest(
                "skops no está instalado: la prueba integral se ejecuta "
                "en el .venv de los estudiantes con MLflow/skops"
            )

        _iris, X_train, _X_test, y_train, _y_test = preparar_iris(0.20, 42)
        for config_nombre, definicion in self.config["modelos"].items():
            with self.subTest(configuracion=config_nombre):
                pipeline = construir_pipeline(definicion)
                pipeline.fit(X_train, y_train)
                with tempfile.TemporaryDirectory() as carpeta:
                    ruta = Path(carpeta) / "modelo.skops"
                    skops.io.dump(pipeline, ruta)
                    recuperado = skops.io.load(
                        ruta,
                        trusted=tipos_confiables_skops(definicion["algoritmo"]),
                    )
                    self.assertEqual(
                        recuperado.predict(X_train[:5]).tolist(),
                        pipeline.predict(X_train[:5]).tolist(),
                    )


if __name__ == "__main__":
    unittest.main()
