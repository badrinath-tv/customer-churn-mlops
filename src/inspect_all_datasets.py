import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

DATASET_1_PATH = "data/raw/Churn_Modelling.csv.xls"
DATASET_2_PATH = "data/raw/Churn_Dataset_2.csv"
DATASET_3_PATH = "data/raw/churn_dataset_3.csv"


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 80)
print("LOADING DATASETS")
print("=" * 80)


# ------------------------------------------------------------
# DATASET 1
# ------------------------------------------------------------
# Dataset 1 contains each complete CSV row inside one quoted
# value. Therefore, pandas initially reads it as one column.
# We manually split the column using commas.

dataset_1_raw = pd.read_csv(
    DATASET_1_PATH
)

dataset_1 = (
    dataset_1_raw
    .iloc[:, 0]
    .astype(str)
    .str.split(
        ",",
        expand=True
    )
)

dataset_1.columns = (
    dataset_1_raw
    .columns[0]
    .split(",")
)

dataset_1.columns = (
    dataset_1.columns
    .str.strip()
)


# ------------------------------------------------------------
# DATASET 2
# ------------------------------------------------------------

dataset_2 = pd.read_csv(
    DATASET_2_PATH
)


# ------------------------------------------------------------
# NORMALIZE DATASET 1 AND DATASET 2
# ------------------------------------------------------------
# Dataset 1 currently contains string values because it was
# manually split.
#
# Dataset 2 contains proper integer, float and object types.
#
# We convert Dataset 1 columns into the corresponding Dataset 2
# data types so that both datasets can be compared correctly.

for column in dataset_1.columns:

    if column in dataset_2.columns:

        try:

            dataset_1[column] = (
                dataset_1[column]
                .astype(
                    dataset_2[column].dtype
                )
            )

        except (ValueError, TypeError):

            dataset_1[column] = (
                dataset_1[column]
                .astype(str)
                .str.strip()
            )

            dataset_2[column] = (
                dataset_2[column]
                .astype(str)
                .str.strip()
            )


# ------------------------------------------------------------
# DATASET 3
# ------------------------------------------------------------

dataset_3 = pd.read_csv(
    DATASET_3_PATH
)


print("All datasets loaded successfully.")


# ============================================================
# DATASET SHAPES
# ============================================================

print("\n" + "=" * 80)
print("DATASET SHAPES")
print("=" * 80)

print(
    "Dataset 1:",
    dataset_1.shape
)

print(
    "Dataset 2:",
    dataset_2.shape
)

print(
    "Dataset 3:",
    dataset_3.shape
)


# ============================================================
# DATASET 1 AND DATASET 2 COLUMN COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("DATASET 1 VS DATASET 2")
print("=" * 80)

same_shape = (
    dataset_1.shape
    ==
    dataset_2.shape
)

same_columns = (
    dataset_1.columns.tolist()
    ==
    dataset_2.columns.tolist()
)

print(
    "Same shape   :",
    same_shape
)

print(
    "Same columns :",
    same_columns
)


# ============================================================
# DATASET 1 AND DATASET 2 EXACT COMPARISON
# ============================================================

if same_shape and same_columns:

    exactly_identical = (
        dataset_1.equals(
            dataset_2
        )
    )

    row_comparison = (
        dataset_1
        .eq(dataset_2)
        .all(axis=1)
    )

    matching_rows = (
        row_comparison.sum()
    )

    different_rows = (
        len(dataset_1)
        -
        matching_rows
    )

    identical_percentage = (
        matching_rows
        /
        len(dataset_1)
    ) * 100

    print(
        "Exactly identical:",
        exactly_identical
    )

    print(
        "Exactly matching rows:",
        matching_rows
    )

    print(
        "Different rows:",
        different_rows
    )

    print(
        "Total rows:",
        len(dataset_1)
    )

    print(
        f"Identical-row percentage: "
        f"{identical_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # FINAL DUPLICATE RESULT
    # --------------------------------------------------------

    print("\n" + "-" * 80)

    if exactly_identical:

        print(
            "RESULT: DATASET 1 AND DATASET 2 "
            "ARE EXACT DUPLICATES."
        )

        print(
            "Dataset 2 should not be considered "
            "an independent dataset."
        )

    elif identical_percentage >= 95:

        print(
            "RESULT: DATASET 1 AND DATASET 2 "
            "ARE NEAR DUPLICATES."
        )

        print(
            "They contain at least 95% "
            "identical rows."
        )

    else:

        print(
            "RESULT: DATASET 1 AND DATASET 2 "
            "ARE DIFFERENT DATASETS."
        )

    print("-" * 80)


else:

    print(
        "The datasets have different "
        "shapes or columns."
    )

    print(
        "An exact row-by-row comparison "
        "cannot be performed."
    )


# ============================================================
# DATASET 1 TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("DATASET 1 TARGET DISTRIBUTION")
print("=" * 80)

if "Exited" in dataset_1.columns:

    dataset_1_target_count = (
        dataset_1["Exited"]
        .value_counts(
            dropna=False
        )
    )

    dataset_1_target_percentage = (
        dataset_1["Exited"]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .mul(100)
        .round(2)
    )

    print(
        dataset_1_target_count
    )

    print("\nPercentage:")

    print(
        dataset_1_target_percentage
    )

else:

    print(
        "Target column 'Exited' "
        "was not found."
    )


# ============================================================
# DATASET 2 TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("DATASET 2 TARGET DISTRIBUTION")
print("=" * 80)

if "Exited" in dataset_2.columns:

    dataset_2_target_count = (
        dataset_2["Exited"]
        .value_counts(
            dropna=False
        )
    )

    dataset_2_target_percentage = (
        dataset_2["Exited"]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .mul(100)
        .round(2)
    )

    print(
        dataset_2_target_count
    )

    print("\nPercentage:")

    print(
        dataset_2_target_percentage
    )

else:

    print(
        "Target column 'Exited' "
        "was not found."
    )


# ============================================================
# DATASET 3 COLUMN NAMES
# ============================================================

print("\n" + "=" * 80)
print("DATASET 3 COLUMNS")
print("=" * 80)

for number, column in enumerate(
    dataset_3.columns,
    start=1
):

    print(
        f"{number:02d}. {column}"
    )


# ============================================================
# DATASET 3 DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("DATASET 3 DATA TYPES")
print("=" * 80)

print(
    dataset_3.dtypes
)


# ============================================================
# DATASET 3 MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("DATASET 3 MISSING VALUES")
print("=" * 80)

dataset_3_missing_values = (
    dataset_3
    .isnull()
    .sum()
)

columns_with_missing_values = (
    dataset_3_missing_values[
        dataset_3_missing_values > 0
    ]
)

if len(
    columns_with_missing_values
) == 0:

    print(
        "No missing values found."
    )

else:

    print(
        columns_with_missing_values
    )

    print(
        "\nTotal missing values:",
        dataset_3_missing_values.sum()
    )


# ============================================================
# DATASET 3 TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("DATASET 3 TARGET DISTRIBUTION")
print("=" * 80)

if "exit" in dataset_3.columns:

    dataset_3_target_count = (
        dataset_3["exit"]
        .value_counts(
            dropna=False
        )
    )

    dataset_3_target_percentage = (
        dataset_3["exit"]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .mul(100)
        .round(2)
    )

    print(
        dataset_3_target_count
    )

    print("\nPercentage:")

    print(
        dataset_3_target_percentage
    )

else:

    print(
        "Target column 'exit' "
        "was not found."
    )


# ============================================================
# DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)

print(
    "Dataset 1:",
    dataset_1
    .duplicated()
    .sum()
)

print(
    "Dataset 2:",
    dataset_2
    .duplicated()
    .sum()
)

print(
    "Dataset 3:",
    dataset_3
    .duplicated()
    .sum()
)


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(
    f"Dataset 1: "
    f"{dataset_1.shape[0]} rows, "
    f"{dataset_1.shape[1]} columns"
)

print(
    f"Dataset 2: "
    f"{dataset_2.shape[0]} rows, "
    f"{dataset_2.shape[1]} columns"
)

print(
    f"Dataset 3: "
    f"{dataset_3.shape[0]} rows, "
    f"{dataset_3.shape[1]} columns"
)


# ============================================================
# INSPECTION COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)