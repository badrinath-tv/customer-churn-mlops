import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATASET_NAME = "Dataset_1"

DATASET_PATH = "data/raw/Churn_Modelling.csv.xls"

RESULTS_DIR = "results/dataset_1"

MODELS_DIR = "models/benchmark_models/dataset_1"

TARGET_COLUMN = "Exited"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD AND PREPARE DATASET
# ============================================================

def load_and_prepare_data():

    print("\n" + "=" * 75)
    print("LOADING DATASET 1")
    print("=" * 75)

    # The file has an .xls extension,
    # but its actual content is comma-separated CSV data.

    dataset = pd.read_csv(
        DATASET_PATH
    )

    # Remove unwanted spaces from column names.

    dataset.columns = (
        dataset.columns
        .str.strip()
    )

    print(
        "Original shape:",
        dataset.shape
    )

    print(
        "Target column:",
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # VERIFY DATASET FORMAT
    # --------------------------------------------------------

    if dataset.shape[1] == 1:

        raise ValueError(
            "\nDataset was loaded as only one column.\n"
            "Detected column:\n"
            f"{dataset.columns.tolist()}\n"
            "The delimiter could not be detected correctly."
        )

    # --------------------------------------------------------
    # VERIFY TARGET COLUMN
    # --------------------------------------------------------

    if TARGET_COLUMN not in dataset.columns:

        raise ValueError(
            f"\nTarget column '{TARGET_COLUMN}' was not found.\n"
            f"Detected columns:\n"
            f"{dataset.columns.tolist()}"
        )

    # --------------------------------------------------------
    # REMOVE IDENTIFIER COLUMNS
    # --------------------------------------------------------

    identifier_columns = [
        "RowNumber",
        "CustomerId",
        "Surname"
    ]

    dataset = dataset.drop(
        columns=identifier_columns,
        errors="ignore"
    )

    print(
        "\nRemoved identifier columns:"
    )

    print(
        identifier_columns
    )

    # --------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    X = dataset.drop(
        columns=[
            TARGET_COLUMN
        ]
    )

    y = (
        dataset[
            TARGET_COLUMN
        ]
        .astype(
            int
        )
    )

    print(
        "\nFinal feature shape:",
        X.shape
    )

    print(
        "\nFinal feature columns:"
    )

    for number, column in enumerate(
        X.columns,
        start=1
    ):

        print(
            f"{number:02d}. {column}"
        )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        "\nTarget percentage:"
    )

    print(
        (
            y.value_counts(
                normalize=True
            )
            *
            100
        ).round(
            2
        )
    )

    return X, y


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

def split_dataset(
    X,
    y
):

    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print(
        "\nTraining samples:",
        len(
            X_train
        )
    )

    print(
        "Testing samples:",
        len(
            X_test
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor():

    numerical_features = [

        "CreditScore",

        "Age",

        "Tenure",

        "Balance",

        "NumOfProducts",

        "HasCrCard",

        "IsActiveMember",

        "EstimatedSalary"
    ]

    categorical_features = [

        "Geography",

        "Gender"
    ]

    # --------------------------------------------------------
    # NUMERICAL PREPROCESSING
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # CATEGORICAL PREPROCESSING
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                OneHotEncoder(

                    handle_unknown="ignore",

                    drop="first",

                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # COMBINE PREPROCESSING PIPELINES
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",

                numerical_pipeline,

                numerical_features
            ),

            (
                "categorical",

                categorical_pipeline,

                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# ORIGINAL MODELS
# ============================================================

def get_original_models():

    return {

        "LogisticRegression":

            LogisticRegression(

                max_iter=2000,

                random_state=RANDOM_STATE
            ),

        "DecisionTree":

            DecisionTreeClassifier(

                random_state=RANDOM_STATE
            ),

        "RandomForest":

            RandomForestClassifier(

                n_estimators=200,

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),

        "XGBoost":

            XGBClassifier(

                n_estimators=200,

                tree_method="hist",

                device="cuda",

                eval_metric="logloss",

                random_state=RANDOM_STATE
            ),

        "LightGBM":

            LGBMClassifier(

                n_estimators=200,

                random_state=RANDOM_STATE,

                n_jobs=-1,

                verbose=-1
            ),

        "CatBoost":

            CatBoostClassifier(

                iterations=200,

                task_type="GPU",

                devices="0",

                random_state=RANDOM_STATE,

                verbose=0
            ),

        "KNN":

            KNeighborsClassifier(

                n_neighbors=5,

                n_jobs=-1
            ),

        "NaiveBayes":

            GaussianNB()
    }


# ============================================================
# CLASS-WEIGHT MODELS
# ============================================================

def get_classweight_models(
    scale_positive_weight
):

    return {

        "LogisticRegression":

            LogisticRegression(

                max_iter=2000,

                class_weight="balanced",

                random_state=RANDOM_STATE
            ),

        "DecisionTree":

            DecisionTreeClassifier(

                class_weight="balanced",

                random_state=RANDOM_STATE
            ),

        "RandomForest":

            RandomForestClassifier(

                n_estimators=200,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),

        "XGBoost":

            XGBClassifier(

                n_estimators=200,

                tree_method="hist",

                device="cuda",

                scale_pos_weight=scale_positive_weight,

                eval_metric="logloss",

                random_state=RANDOM_STATE
            ),

        "LightGBM":

            LGBMClassifier(

                n_estimators=200,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1,

                verbose=-1
            ),

        "CatBoost":

            CatBoostClassifier(

                iterations=200,

                class_weights=[

                    1,

                    scale_positive_weight
                ],

                task_type="GPU",

                devices="0",

                random_state=RANDOM_STATE,

                verbose=0
            )
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (

            model.predict_proba(
                X_test
            )[
                :,
                1
            ]
        )

    elif hasattr(
        model,
        "decision_function"
    ):

        probabilities = (

            model.decision_function(
                X_test
            )
        )

    else:

        probabilities = predictions

    return {

        "Accuracy":

            accuracy_score(

                y_test,

                predictions
            ),

        "Precision":

            precision_score(

                y_test,

                predictions,

                zero_division=0
            ),

        "Recall":

            recall_score(

                y_test,

                predictions,

                zero_division=0
            ),

        "F1":

            f1_score(

                y_test,

                predictions,

                zero_division=0
            ),

        "ROC_AUC":

            roc_auc_score(

                y_test,

                probabilities
            )
    }


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

def save_model(
    model,
    model_name,
    strategy
):

    safe_model_name = (

        model_name

        .replace(
            " ",
            "_"
        )

        .lower()
    )

    safe_strategy_name = (

        strategy

        .replace(
            " ",
            "_"
        )

        .lower()
    )

    model_filename = (

        f"{safe_strategy_name}"

        f"__"

        f"{safe_model_name}"

        f".pkl"
    )

    model_path = os.path.join(

        MODELS_DIR,

        model_filename
    )

    joblib.dump(

        model,

        model_path
    )


# ============================================================
# RUN MODELS
# ============================================================

def run_models(
    strategy,
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    results
):

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        f"RUNNING {strategy.upper()} EXPERIMENTS"
    )

    print(
        "=" * 75
    )

    total_models = len(
        models
    )

    for model_number, (
        model_name,
        model

    ) in enumerate(

        models.items(),

        start=1
    ):

        print(

            f"\n"

            f"[{model_number}/"

            f"{total_models}] "

            f"{strategy} + "

            f"{model_name}"
        )

        start_time = time.time()

        try:

            model.fit(

                X_train,

                y_train
            )

            metrics = evaluate_model(

                model,

                X_test,

                y_test
            )

            training_time = (

                time.time()

                -

                start_time
            )

            results.append(

                {

                    "Dataset":

                        DATASET_NAME,

                    "Strategy":

                        strategy,

                    "Model":

                        model_name,

                    "Status":

                        "Completed",

                    **metrics,

                    "Train_Time":

                        round(

                            training_time,

                            2
                        ),

                    "Error":

                        np.nan
                }
            )

            save_model(

                model,

                model_name,

                strategy
            )

            print(

                f"Accuracy : "

                f"{metrics['Accuracy']:.4f}"
            )

            print(

                f"Precision: "

                f"{metrics['Precision']:.4f}"
            )

            print(

                f"Recall   : "

                f"{metrics['Recall']:.4f}"
            )

            print(

                f"F1 Score : "

                f"{metrics['F1']:.4f}"
            )

            print(

                f"ROC AUC  : "

                f"{metrics['ROC_AUC']:.4f}"
            )

            print(

                f"Time     : "

                f"{training_time:.2f} seconds"
            )

        except Exception as error:

            training_time = (

                time.time()

                -

                start_time
            )

            results.append(

                {

                    "Dataset":

                        DATASET_NAME,

                    "Strategy":

                        strategy,

                    "Model":

                        model_name,

                    "Status":

                        "Failed",

                    "Accuracy":

                        np.nan,

                    "Precision":

                        np.nan,

                    "Recall":

                        np.nan,

                    "F1":

                        np.nan,

                    "ROC_AUC":

                        np.nan,

                    "Train_Time":

                        round(

                            training_time,

                            2
                        ),

                    "Error":

                        str(
                            error
                        )
                }
            )

            print(
                "FAILED:"
            )

            print(
                error
            )


# ============================================================
# ADD NOT-APPLICABLE CLASS-WEIGHT COMBINATIONS
# ============================================================

def add_not_applicable_results(
    results
):

    unsupported_models = [

        "KNN",

        "NaiveBayes"
    ]

    for model_name in unsupported_models:

        results.append(

            {

                "Dataset":

                    DATASET_NAME,

                "Strategy":

                    "ClassWeight",

                "Model":

                    model_name,

                "Status":

                    "Not Applicable",

                "Accuracy":

                    np.nan,

                "Precision":

                    np.nan,

                "Recall":

                    np.nan,

                "F1":

                    np.nan,

                "ROC_AUC":

                    np.nan,

                "Train_Time":

                    np.nan,

                "Error":

                    (
                        "Native class weighting "
                        "is not supported."
                    )
            }
        )


# ============================================================
# SAVE RESULT FILES
# ============================================================

def save_results(
    results
):

    results_dataframe = pd.DataFrame(
        results
    )

    required_columns = [

        "Dataset",

        "Strategy",

        "Model",

        "Status",

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC",

        "Train_Time",

        "Error"
    ]

    results_dataframe = (

        results_dataframe[

            required_columns

        ]
    )

    # --------------------------------------------------------
    # SAVE ALL RESULTS
    # --------------------------------------------------------

    full_results_path = os.path.join(

        RESULTS_DIR,

        "full_benchmark_results.csv"
    )

    results_dataframe.to_csv(

        full_results_path,

        index=False
    )

    # --------------------------------------------------------
    # CREATE RANKING USING F1 SCORE
    # --------------------------------------------------------

    ranking_dataframe = (

        results_dataframe

        [

            results_dataframe[

                "Status"

            ]

            ==

            "Completed"
        ]

        .dropna(

            subset=[

                "F1"
            ]
        )

        .sort_values(

            by=[

                "F1",

                "ROC_AUC"
            ],

            ascending=[

                False,

                False
            ]
        )

        .reset_index(

            drop=True
        )
    )

    ranking_dataframe.index = (

        ranking_dataframe.index

        +

        1
    )

    ranking_dataframe.index.name = "Rank"

    ranking_path = os.path.join(

        RESULTS_DIR,

        "model_ranking.csv"
    )

    ranking_dataframe.to_csv(

        ranking_path,

        index=True
    )

    # --------------------------------------------------------
    # SAVE TOP FIVE MODELS
    # --------------------------------------------------------

    top_five_dataframe = (

        ranking_dataframe

        .head(
            5
        )
    )

    top_five_path = os.path.join(

        RESULTS_DIR,

        "top_5_models.csv"
    )

    top_five_dataframe.to_csv(

        top_five_path,

        index=True
    )

    return (

        results_dataframe,

        ranking_dataframe
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    total_start_time = time.time()

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "DATASET 1: GPU-ACCELERATED MODEL BENCHMARK"
    )

    print(
        "=" * 75
    )

    print(
        "\nExecution configuration:"
    )

    print(
        "XGBoost : NVIDIA GPU"
    )

    print(
        "CatBoost: NVIDIA GPU"
    )

    print(
        "Other models: CPU"
    )

    print(
        "SVM: Excluded because of computational scalability"
    )

    print(
        "\nExperimental design:"
    )

    print(
        "8 algorithms"
    )

    print(
        "3 imbalance-handling strategies"
    )

    print(
        "24 documented combinations"
    )

    print(
        "22 actual model-training runs"
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    X, y = load_and_prepare_data()

    # --------------------------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = split_dataset(

        X,

        y
    )

    # --------------------------------------------------------
    # PREPROCESS DATASET
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "PREPROCESSING DATASET 1"
    )

    print(
        "=" * 75
    )

    preprocessor = create_preprocessor()

    X_train_processed = (

        preprocessor

        .fit_transform(

            X_train
        )
    )

    X_test_processed = (

        preprocessor

        .transform(

            X_test
        )
    )

    print(

        "Processed training shape:",

        X_train_processed.shape
    )

    print(

        "Processed testing shape:",

        X_test_processed.shape
    )

    # --------------------------------------------------------
    # SAVE PREPROCESSOR
    # --------------------------------------------------------

    preprocessor_path = os.path.join(

        MODELS_DIR,

        "preprocessor.pkl"
    )

    joblib.dump(

        preprocessor,

        preprocessor_path
    )

    print(
        "\nPreprocessor saved:"
    )

    print(
        preprocessor_path
    )

    # --------------------------------------------------------
    # CALCULATE CLASS-WEIGHT RATIO
    # --------------------------------------------------------

    negative_count = (

        y_train

        .value_counts()[0]
    )

    positive_count = (

        y_train

        .value_counts()[1]
    )

    scale_positive_weight = (

        negative_count

        /

        positive_count
    )

    print(
        "\nClass-weight ratio:"
    )

    print(

        round(

            scale_positive_weight,

            4
        )
    )

    results = []

    # --------------------------------------------------------
    # ORIGINAL EXPERIMENTS
    # --------------------------------------------------------

    run_models(

        strategy="Original",

        models=get_original_models(),

        X_train=X_train_processed,

        y_train=y_train,

        X_test=X_test_processed,

        y_test=y_test,

        results=results
    )

    # --------------------------------------------------------
    # CLASS-WEIGHT EXPERIMENTS
    # --------------------------------------------------------

    run_models(

        strategy="ClassWeight",

        models=get_classweight_models(

            scale_positive_weight
        ),

        X_train=X_train_processed,

        y_train=y_train,

        X_test=X_test_processed,

        y_test=y_test,

        results=results
    )

    # --------------------------------------------------------
    # RECORD UNSUPPORTED CLASS-WEIGHT COMBINATIONS
    # --------------------------------------------------------

    add_not_applicable_results(
        results
    )

    # --------------------------------------------------------
    # APPLY SMOTE ONLY TO TRAINING DATA
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "APPLYING SMOTE TO TRAINING DATA ONLY"
    )

    print(
        "=" * 75
    )

    smote = SMOTE(

        random_state=RANDOM_STATE
    )

    (
        X_train_smote,
        y_train_smote

    ) = smote.fit_resample(

        X_train_processed,

        y_train
    )

    print(
        "Before SMOTE:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nAfter SMOTE:"
    )

    print(

        pd.Series(

            y_train_smote

        ).value_counts()
    )

    # --------------------------------------------------------
    # SMOTE EXPERIMENTS
    # --------------------------------------------------------

    run_models(

        strategy="SMOTE",

        models=get_original_models(),

        X_train=X_train_smote,

        y_train=y_train_smote,

        X_test=X_test_processed,

        y_test=y_test,

        results=results
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    (
        results_dataframe,
        ranking_dataframe

    ) = save_results(

        results
    )

    # --------------------------------------------------------
    # FINAL EXECUTION SUMMARY
    # --------------------------------------------------------

    total_time = (

        time.time()

        -

        total_start_time
    )

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "DATASET 1 BENCHMARK COMPLETED"
    )

    print(
        "=" * 75
    )

    print(

        "\nDocumented combinations:",

        len(

            results_dataframe
        )
    )

    completed_count = (

        results_dataframe[

            "Status"

        ]

        .eq(

            "Completed"
        )

        .sum()
    )

    print(

        "Completed experiments:",

        completed_count
    )

    failed_count = (

        results_dataframe[

            "Status"

        ]

        .eq(

            "Failed"
        )

        .sum()
    )

    print(

        "Failed experiments:",

        failed_count
    )

    not_applicable_count = (

        results_dataframe[

            "Status"

        ]

        .eq(

            "Not Applicable"
        )

        .sum()
    )

    print(

        "Not-applicable combinations:",

        not_applicable_count
    )

    print(

        "Total benchmark time:",

        f"{total_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # DISPLAY TOP FIVE MODELS
    # --------------------------------------------------------

    print(
        "\nTOP 5 MODELS"
    )

    print(
        "-" * 75
    )

    print(

        ranking_dataframe[

            [

                "Strategy",

                "Model",

                "Accuracy",

                "Precision",

                "Recall",

                "F1",

                "ROC_AUC",

                "Train_Time"
            ]

        ]

        .head(
            5
        )
    )

    # --------------------------------------------------------
    # DISPLAY WINNING MODEL
    # --------------------------------------------------------

    if not ranking_dataframe.empty:

        winner = (

            ranking_dataframe

            .iloc[0]
        )

        print(
            "\n"
            +
            "=" * 75
        )

        print(
            "DATASET 1 WINNER"
        )

        print(
            "=" * 75
        )

        print(

            "Strategy :",

            winner[
                "Strategy"
            ]
        )

        print(

            "Model    :",

            winner[
                "Model"
            ]
        )

        print(

            "Accuracy :",

            f"{winner['Accuracy']:.4f}"
        )

        print(

            "Precision:",

            f"{winner['Precision']:.4f}"
        )

        print(

            "Recall   :",

            f"{winner['Recall']:.4f}"
        )

        print(

            "F1 Score :",

            f"{winner['F1']:.4f}"
        )

        print(

            "ROC AUC  :",

            f"{winner['ROC_AUC']:.4f}"
        )

    # --------------------------------------------------------
    # DISPLAY OUTPUT FILE PATHS
    # --------------------------------------------------------

    print(
        "\nRESULT FILES"
    )

    print(
        "-" * 75
    )

    print(

        os.path.join(

            RESULTS_DIR,

            "full_benchmark_results.csv"
        )
    )

    print(

        os.path.join(

            RESULTS_DIR,

            "model_ranking.csv"
        )
    )

    print(

        os.path.join(

            RESULTS_DIR,

            "top_5_models.csv"
        )
    )


# ============================================================
# EXECUTE PROGRAM
# ============================================================

if __name__ == "__main__":

    main()