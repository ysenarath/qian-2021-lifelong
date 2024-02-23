import csv
from pathlib import Path
from typing import Literal, Set
import pandas as pd

pwd = Path(".").absolute()
version = "v1"


def create_dataset_fold(
    df: pd.DataFrame,
    fold: Literal["train", "valid", "test"],
    target_group_set: Set[str],
    filename: str = None,
) -> str:
    out_df = df["text"].to_frame()
    target_group = df["_source"].apply(pd.Series)["target_groups"].explode().unique()[0]
    out_df["pos_group"] = df["labels"].map({False: target_group, True: ""})
    out_df["neg_group"] = "&&".join(target_group_set - {target_group}) + df[
        "labels"
    ].map({False: "", True: f"&&{target_group}"})
    output_path = (
        pwd / "data" / "processed" / f"qian-2021-lifelong-{version}" / fold / filename
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    # validate that the file was written correctly
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pass


def get_filename(df: pd.DataFrame) -> str:
    groups_df = df["groups"].apply(pd.Series)
    current_group_name = groups_df.columns[groups_df.sum(axis=0) > 0][0]
    return f"{current_group_name}.csv"


def create_dataset(data_df):
    target_group_set = set()
    step_train_df: pd.DataFrame = None
    for idx, (step_train_df, step_valid_df, test_dfs) in data_df:
        target_group = (
            step_train_df["_source"]
            .apply(pd.Series)["target_groups"]
            .explode()
            .unique()[0]
        )
        target_group_set.add(target_group)
    order = []
    for idx, (step_train_df, step_valid_df, test_dfs) in data_df:
        filename = get_filename(step_train_df)
        create_dataset_fold(
            step_train_df,
            "train",
            target_group_set,
            filename,
        )
        create_dataset_fold(
            step_valid_df,
            "valid",
            target_group_set,
            filename,
        )
        order.append(filename)
    test_filenames = set()
    for _, test_df in test_dfs:
        filename = get_filename(test_df)
        test_filenames.add(filename)
        if filename not in order:
            msg = f"test target '{filename}' not found in train/valid"
            print("Warning:", msg)
            continue
        create_dataset_fold(test_df, "test", target_group_set, filename)
    if set(order).difference(test_filenames):
        msg = "some train/valid targets not found in test"
        raise ValueError(msg)
    with open(
        pwd / "data" / "processed" / f"qian-2021-lifelong-{version}" / "order.txt", "w"
    ) as file:
        for line in order:
            file.write(line + "\n")
