from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True

except ImportError:
    SMOTE = None
    ADASYN = None
    SMOTEENN = None
    IMBLEARN_AVAILABLE = False

DEFAULT_RANDOM_STATE = 42
# modeling_utils.py está dentro de TG2026/src/,
# por lo que parents[1] corresponde a TG2026/
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# PREPROCESAMIENTO
# ============================================================

def make_one_hot_encoder():
    """
    OneHotEncoder compatible con distintas versiones de sklearn.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32,
        )


def safe_log1p_array(X):
    """
    Aplica log1p a variables numéricas no negativas. Se usa np.maximum para evitar errores si aparece algún valor negativo inesperado.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0)
    return np.log1p(X).astype(np.float32)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Codifica variables categóricas usando frecuencia relativa aprendida únicamente durante fit.
    Categorías no vistas posteriormente se codifican como 0.
    """
    def __init__(self, columns=None, suffix="_freq"):
        self.columns = columns
        self.suffix = suffix

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = list(self.columns)

        self.frequency_maps_ = {}

        for col in self.columns_:
            values = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
            )

            freq = values.value_counts(
                normalize=True
            )

            self.frequency_maps_[col] = (
                freq.to_dict()
            )

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        output = pd.DataFrame(
            index=X.index
        )

        for col in self.columns_:
            values = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
            )

            output[f"{col}{self.suffix}"] = (
                values
                .map(self.frequency_maps_[col])
                .fillna(0)
                .astype(np.float32)
            )

        return output

    def get_feature_names_out(
        self,
        input_features=None
    ):
        return np.array(
            [
                f"{col}{self.suffix}"
                for col in self.columns_
            ],
            dtype=object,
        )


class IPFeatureEncoder(
    BaseEstimator,
    TransformerMixin
):
    """
    Codificación compacta para columnas IP. Para cada IP genera:
    - frecuencia de la IP exacta en train
    - frecuencia del prefijo en train
    - indicador de IPv6
    """
    def __init__(
        self,
        columns=None,
        prefix_suffix="_prefix_freq",
        ip_suffix="_ip_freq",
    ):
        self.columns = columns
        self.prefix_suffix = prefix_suffix
        self.ip_suffix = ip_suffix


    def _get_prefix(self, series):
        s = (
            series
            .astype("string")
            .fillna("__MISSING__")
        )

        is_ipv6 = s.str.contains(
            ":",
            regex=False,
            na=False
        )

        prefix = pd.Series(
            "__UNKNOWN__",
            index=s.index,
            dtype="string",
        )

        # IPv4: primeros tres octetos,
        # prefijo aproximado /24
        ipv4_values = s[~is_ipv6]
        ipv4_prefix = (
            ipv4_values
            .str.extract(
                r"^(\d+\.\d+\.\d+)",
                expand=False
            )
            .fillna("__UNKNOWN__")
        )
        prefix.loc[~is_ipv6] = (ipv4_prefix)

        # IPv6: primeros cuatro bloques
        ipv6_values = s[is_ipv6]
        ipv6_prefix = (
            ipv6_values
            .str.split(":")
            .str[:4]
            .str.join(":")
            .fillna("__UNKNOWN__")
        )
        prefix.loc[is_ipv6] = (ipv6_prefix)
        return prefix.astype("string")


    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = list(self.columns)

        self.ip_frequency_maps_ = {}
        self.prefix_frequency_maps_ = {}
        for col in self.columns_:
            values = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
            )
            prefixes = self._get_prefix(values)
            self.ip_frequency_maps_[col] = (
                values
                .value_counts(normalize=True)
                .to_dict()
            )
            self.prefix_frequency_maps_[col] = (
                prefixes
                .value_counts(normalize=True)
                .to_dict()
            )

        return self


    def transform(self, X):
        X = pd.DataFrame(X).copy()
        output = pd.DataFrame(index=X.index)
        for col in self.columns_:
            values = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
            )
            prefixes = self._get_prefix(values)
            is_ipv6 = values.str.contains(
                ":",
                regex=False,
                na=False
            )
            output[f"{col}{self.ip_suffix}"] = (
                values
                .map(
                    self.ip_frequency_maps_[col]
                )
                .fillna(0)
                .astype(np.float32)
            )

            output[
                f"{col}{self.prefix_suffix}"
            ] = (
                prefixes
                .map(
                    self.prefix_frequency_maps_[col]
                )
                .fillna(0)
                .astype(np.float32)
            )

            output[
                f"{col}_is_ipv6"
            ] = is_ipv6.astype(np.int8)

        return output

    def get_feature_names_out(
        self,
        input_features=None
    ):
        names = []
        for col in self.columns_:
            names.extend([
                f"{col}{self.ip_suffix}",
                f"{col}{self.prefix_suffix}",
                f"{col}_is_ipv6",
            ])

        return np.array(names, dtype=object)


class PortFeatureEncoder(
    BaseEstimator,
    TransformerMixin
):
    """
    Genera variables binarias para los puertos:
    - puerto cero
    - well-known: 1-1023
    - registrado: 1024-49151
    - dinámico/efímero: 49152-65535
    """
    def __init__( self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = list(
                self.columns
            )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        output = pd.DataFrame(index=X.index)
        for col in self.columns_:
            values = pd.to_numeric(
                X[col],
                errors="coerce"
            ).fillna(-1)

            output[
                f"{col}_is_zero"
            ] = (
                values.eq(0)
                .astype(np.int8)
            )

            output[
                f"{col}_is_well_known"
            ] = (
                values
                .between(1, 1023)
                .astype(np.int8)
            )

            output[
                f"{col}_is_registered"
            ] = (
                values
                .between(1024, 49151)
                .astype(np.int8)
            )

            output[
                f"{col}_is_dynamic"
            ] = (
                values
                .between(49152, 65535)
                .astype(np.int8)
            )

        return output

    def get_feature_names_out(
        self,
        input_features=None
    ):
        names = []
        for col in self.columns_:
            names.extend([
                f"{col}_is_zero",
                f"{col}_is_well_known",
                f"{col}_is_registered",
                f"{col}_is_dynamic",
            ])

        return np.array(
            names,
            dtype=object
        )


def build_base_preprocessor(
    preprocessing_config,
    model_family="linear"
):
    """
    Construye el preprocesador base común.
    model_family:
        - linear
        - ae
        - isolation_forest
        - xgboost
    Las columnas se leen de preprocessing_config, evitando depender de variables globales del notebook.
    """
    valid_model_families = {
        "linear",
        "ae",
        "isolation_forest",
        "xgboost",
    }

    if model_family not in valid_model_families:
        raise ValueError(
            "model_family debe estar en "
            f"{valid_model_families}"
        )

    numeric_log_columns = (
        preprocessing_config[
            "numeric_log_columns"
        ]
    )

    binary_flag_columns = (
        preprocessing_config[
            "binary_flag_columns"
        ]
    )

    one_hot_columns = (
        preprocessing_config[
            "one_hot_columns"
        ]
    )

    frequency_encoding_columns = (
        preprocessing_config[
            "frequency_encoding_columns"
        ]
    )

    ip_columns = (
        preprocessing_config[
            "ip_columns"
        ]
    )

    port_feature_columns = (
        preprocessing_config[
            "port_feature_columns"
        ]
    )

    scale_numeric = model_family in {
        "linear",
        "ae",
        "isolation_forest",
    }

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "log1p",
            FunctionTransformer(
                safe_log1p_array,
                validate=False,
                feature_names_out="one-to-one",
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    numeric_log_pipeline = Pipeline(
        steps=numeric_steps
    )

    one_hot_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num_log",
                numeric_log_pipeline,
                numeric_log_columns,
            ),
            (
                "flags",
                "passthrough",
                binary_flag_columns,
            ),
            (
                "onehot",
                one_hot_pipeline,
                one_hot_columns,
            ),
            (
                "frequency",
                FrequencyEncoder(
                    columns=frequency_encoding_columns
                ),
                frequency_encoding_columns,
            ),
            (
                "ip_features",
                IPFeatureEncoder(
                    columns=ip_columns
                ),
                ip_columns,
            ),
            (
                "port_features",
                PortFeatureEncoder(
                    columns=port_feature_columns
                ),
                port_feature_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_transformed_feature_names(preprocessor):
    """
    Recupera los nombres de variables generadas por un preprocesador ya ajustado.
    """
    try:
        return (preprocessor.get_feature_names_out())

    except Exception as error:
        print("No fue posible obtener los nombres de variables transformadas.")
        print(error)

        return None

# ============================================================
# CARGA DE DATOS PARA MODELADO
# ============================================================
def quote_identifier(column_name):
    return (
        '"'
        + column_name.replace('"', '""')
        + '"'
    )

def get_modeling_dataset_path(
    use_dev_sample=True,
    project_root=None,
):
    """
    Retorna la ruta del dataset a utilizar.
    """
    if project_root is None:
        project_root = (DEFAULT_PROJECT_ROOT)

    project_root = Path(project_root)
    interim_path = (project_root / "data" / "interim")
    if use_dev_sample:
        return (interim_path / "dataset_modelado_dev.parquet")

    return (interim_path / "dataset_modelado_con_split.parquet")


def get_modeling_sql_path(
    use_dev_sample=True,
    project_root=None,
):
    dataset_path = (get_modeling_dataset_path(use_dev_sample=use_dev_sample, project_root=project_root))
    if not dataset_path.exists():
        raise FileNotFoundError("No existe el dataset solicitado: " f"{dataset_path}")

    return (
        dataset_path
        .resolve()
        .as_posix()
        .replace("'", "''")
    )


def load_split_dataframe(
    split_name,
    preprocessing_config,
    use_dev_sample=True,
    columns=None,
    limit=None,
    project_root=None,
):
    """
    Carga un split específico en pandas.
    split_name: train, valid o test
    use_dev_sample:
        True  -> dataset_modelado_dev.parquet
        False -> dataset_modelado_con_split.parquet
    """

    if split_name not in {
        "train",
        "valid",
        "test",
    }:
        raise ValueError("split_name debe ser 'train', 'valid' o 'test'.")

    dataset_sql_path = (get_modeling_sql_path(use_dev_sample=use_dev_sample, project_root=project_root))
    if columns is None:
        columns = (
            preprocessing_config[
                "predictor_candidate_columns"
            ]
            + preprocessing_config[
                "target_columns"
            ]
            + preprocessing_config[
                "auxiliary_columns"
            ]
            + [
                "clase",
                "strata_key",
                "split",
                "row_id_model",
            ]
        )
    # Evitar columnas duplicadas
    columns = list(dict.fromkeys(columns))
    columns_sql = ",\n        ".join(
        quote_identifier(col)
        for col in columns
    )
    limit_sql = ""
    if limit is not None:
        limit_sql = (f"\nLIMIT {int(limit)}")

    query = f"""
    SELECT {columns_sql}
    FROM read_parquet('{dataset_sql_path}')
    WHERE split = '{split_name}'
    {limit_sql}
    """
    connection = duckdb.connect()
    try:
        df = (connection.execute(query).df())

    finally:
        connection.close()

    return df


def split_features_target(
    df,
    preprocessing_config,
    task="binary",
):
    """
    Separa X e y.
    binary: usa attack_a.
    multiclass: conserva solamente attack_a = 1 y usa attack_t.
    """
    if task not in {
        "binary",
        "multiclass"
    }:
        raise ValueError("task debe ser 'binary' o 'multiclass'.")

    predictor_columns = (preprocessing_config["predictor_candidate_columns"])
    binary_target_column = (preprocessing_config["binary_target_column"])
    multiclass_target_column = (preprocessing_config["multiclass_target_column"])

    if task == "binary":
        X = df[
            predictor_columns
        ].copy()

        y = df[
            binary_target_column
        ].copy()

    else:
        attack_mask = (
            df[binary_target_column]
            .eq(1)
        )

        X = df.loc[
            attack_mask,
            predictor_columns,
        ].copy()

        y = df.loc[
            attack_mask,
            multiclass_target_column,
        ].copy()

    return X, y


# ============================================================
# DESBALANCE DE CLASES
# ============================================================
def compute_class_weight_dict(y):
    """
    Calcula pesos de clase balanceados para modelos de sklearn.
    """
    y_array = np.asarray(y)
    classes = np.unique(y_array)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_array,
    )
    return dict(zip(classes, weights))


def compute_binary_scale_pos_weight(
    y,
    positive_label=1,
):
    """
    scale_pos_weight para XGBoost: negativos / positivos.
    """
    counts = (pd.Series(y).value_counts())
    n_positive = int(counts.get(positive_label, 0))
    n_negative = int(counts.drop(labels=[positive_label], errors="ignore").sum())
    if n_positive == 0:
        raise ValueError("No hay registros positivos para calcular scale_pos_weight.")

    return (n_negative / n_positive)


def make_binary_sampling_strategy(
    y,
    target_ratio_upper_bound,
    max_minority_multiplier,
    max_synthetic_samples,
    positive_label=1,
):
    """
    Estrategia binaria de sobremuestreo acotado. El objetivo final de la clase positiva es el mínimo entre:
    1. ratio máximo respecto a la clase normal.
    2. crecimiento relativo máximo permitido.
    3. número máximo absoluto de sintéticos.
    """
    counts = (pd.Series(y).value_counts())
    n_positive = int(counts.get(positive_label, 0))
    n_negative = int(counts.drop(labels=[positive_label], errors="ignore").sum())
    
    if (n_positive == 0 or n_negative == 0):
        raise ValueError("La tarea binaria requiere presencia de ambas clases.")

    if not (0 < target_ratio_upper_bound <= 1):
        raise ValueError("target_ratio_upper_bound debe estar en (0, 1].")

    if max_minority_multiplier < 1:
        raise ValueError("max_minority_multiplier debe ser >= 1.")

    if max_synthetic_samples < 0:
        raise ValueError("max_synthetic_samples debe ser >= 0.")

    target_by_ratio = int(
        np.floor(
            target_ratio_upper_bound
            * n_negative
        )
    )

    target_by_multiplier = int(
        np.floor(
            max_minority_multiplier
            * n_positive
        )
    )

    target_by_absolute_cap = int(n_positive + max_synthetic_samples)
    target_count = min(
        target_by_ratio,
        target_by_multiplier,
        target_by_absolute_cap,
    )

    if target_count <= n_positive:
        return None

    return {
        positive_label: target_count
    }


def make_multiclass_sampling_strategy(
    y,
    target_count_upper_bound,
    max_class_multiplier,
    max_synthetic_per_class,
):
    """
    Construye una estrategia de sobremuestreo multiclase acotada. Cada clase minoritaria puede aumentar hasta el mínimo entre:
    - un límite superior absoluto,
    - un multiplicador de su tamaño original,
    - un máximo de muestras sintéticas adicionales.
    """
    counts = pd.Series(y).value_counts()
    sampling_strategy = {}
    for class_label, count in counts.items():
        if count >= target_count_upper_bound:
            continue

        target_by_multiplier = int(np.floor(count * max_class_multiplier))
        target_by_absolute_cap = int(count + max_synthetic_per_class)
        target_count = min(target_count_upper_bound, target_by_multiplier, target_by_absolute_cap)
        if target_count > count:
            sampling_strategy[class_label] = target_count

    return sampling_strategy if sampling_strategy else None


def build_resampler(
    strategy_name,
    task,
    y_train,
    imbalance_config,
    random_state=DEFAULT_RANDOM_STATE,
    binary_profile="conservative",
):
    """
    Construye el remuestreador. none y class_weight retornan None porque no modifican las observaciones.
    """
    valid_strategies = set(imbalance_config["imbalance_strategies"])

    if strategy_name not in valid_strategies:
        raise ValueError("Estrategia no válida: "f"{strategy_name}")

    if task not in {"binary", "multiclass"}:
        raise ValueError("task debe ser 'binary' o 'multiclass'.")

    if strategy_name in {"none", "class_weight"}:
        return None

    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn no está "
            "disponible. Instálalo con: "
            "pip install imbalanced-learn"
        )

    if task == "binary":
        profiles = imbalance_config["binary_oversampling_profiles"]
        if binary_profile not in profiles:
            raise ValueError(
                f"Perfil binario no válido: {binary_profile}. "
                f"Disponibles: {list(profiles)}"
            )
        binary_config = profiles[binary_profile]
        sampling_strategy = make_binary_sampling_strategy(
            y=y_train,
            target_ratio_upper_bound=binary_config["ratio_upper_bound"],
            max_minority_multiplier=binary_config["max_minority_multiplier"],
            max_synthetic_samples=binary_config["max_synthetic_samples"],
            positive_label=1,
        )

    else:
        sampling_strategy = make_multiclass_sampling_strategy(
            y=y_train,
            target_count_upper_bound=imbalance_config[
                "multiclass_target_count_upper_bound"
            ],
            max_class_multiplier=imbalance_config[
                "multiclass_max_class_multiplier"
            ],
            max_synthetic_per_class=imbalance_config[
                "multiclass_max_synthetic_per_class"
            ],
        )

    if sampling_strategy is None:
        return None

    smote_k_neighbors = (
        imbalance_config["smote_k_neighbors"]
    )

    adasyn_n_neighbors = (
        imbalance_config["adasyn_n_neighbors"]
    )
    if strategy_name == "smote":
        return SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=smote_k_neighbors,
        )

    if strategy_name == "adasyn":
        return ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=adasyn_n_neighbors,
        )

    if strategy_name == "smote_enn":
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=smote_k_neighbors,
        )
        return SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            smote=smote,
        )

    raise ValueError("No se pudo construir la estrategia: " f"{strategy_name}")


def validate_imbalance_strategy_for_model(
    strategy_name,
    model_family,
    imbalance_config,
):
    """
    Comprueba la compatibilidad entre modelo y estrategia de desbalance.
    """
    valid_model_families = {
        "logistic_regression",
        "xgboost",
        "ae_lr",
        "isolation_forest",
    }

    if model_family not in valid_model_families:
        raise ValueError("model_family debe estar en " f"{valid_model_families}")

    valid_strategies = set(imbalance_config["imbalance_strategies"])

    if strategy_name not in valid_strategies:
        raise ValueError("Estrategia no válida: " f"{strategy_name}")

    if (model_family == "isolation_forest" and strategy_name != "none"):
        raise ValueError(
            "Las estrategias de desbalance supervisadas no aplican a Isolation Forest. Para este modelo se usará 'none'."
        )
    return True


def summarize_class_distribution(
    y,
    name="dataset",
):
    """
    Resume la distribución de clases de un vector objetivo.
    """
    df_summary = (
        pd.Series(y)
        .value_counts()
        .rename_axis("clase")
        .reset_index(name="n")
    )
    df_summary["porcentaje"] = (
        100
        * df_summary["n"]
        / df_summary["n"].sum()
    )
    df_summary.insert(
        0,
        "dataset",
        name
    )
    return df_summary
