from re import M
import pytest
from src.preprocessing import load_csv, PreprocessPetFinder
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"

#* ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
#*  'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt',
#*  'AdoptionSpeed']

@pytest.fixture
def csv():
    return load_csv(url)


@pytest.fixture
def PPF(csv):
    return PreprocessPetFinder(csv, "AdoptionSpeed", "Description")

def test_set_get(PPF):
    print(PPF.df)
    PPF.df = ["emptyyyyy"]
    print(PPF.df)
    assert 0

def test_ppf_general(PPF):
    # print(PPF.df.head())
    # print(PPF.df.columns)
    # print(PPF.df[["Color1", "Color2", "MaturitySize", "FurLength"]])
    sns.heatmap(PPF.df.corr(), annot=True)
    plt.savefig("corrs.png")
    assert 0

# todo make it flexible
# @pytest.fixture
# def fbk(PPF):  # * stands for FilteredByKey
#     return PPF.filter_by_key("Age", 50)

@pytest.fixture
def train_ds(PPF):
    train_df, _, _ = PPF.split_df()
    train = PPF.df_to_ds(train_df)
    return train


@pytest.fixture
def layers(PPF, train_ds):
    input, encoded = PPF.ds_encode(train_ds)
    return input, encoded


def test_csv_contents(csv):
    print(csv.head())
    assert 0


def test_get_df_colsdata(csv):
    ppf = PreprocessPetFinder(csv, "AdoptionSpeed")
    assert len(ppf.df_colsdata) == 0

def test_binning_feature(PPF):
    PPF.binning_feature("Age")
    assert 0


def test_drop_unused(PPF):
    print(PPF.df.head())
    assert 0


def test_split_df(PPF):
    print(len(PPF.train), len(PPF.val), len(PPF.test))
    assert 0


def test_df_to_ds(PPF):
    train, val, test = PPF.split_df()
    PPF.df_to_ds(train)
    assert 0


def test_ds_encode(PPF, train_ds):
    PPF.ds_encode(train_ds)
    assert 0


def test_numerical_encode(PPF, train_ds):
    norm = PPF.encode_numerical(train_ds, "Age")
    for v, _ in train_ds:
        print(norm(v["Age"]))
    assert 0


def test_categorical_encode(PPF, train_ds):
    encoding_layer = PPF.encode_categorical(train_ds, "Breed1", "string")
    for v, _ in train_ds:
        print(encoding_layer(v["Breed1"]))
    # print(encodeding_layer(train_feature))
    assert 0

def test_get_imbalance(PPF):
    PPF.filter_by_pet_age(84)
    PPF.drop_rows_by_key("Sterilized", "Not Sure")
    PPF.drop_rows_by_key("Health", "Minor injury")
    PPF.drop_rows_by_key("Health", "Serious injury")
    print(PPF.df["AdoptionSpeed"].value_counts())
    assert 0

def test_remove_imbalance_1(PPF):
    print(PPF.df["AdoptionSpeed"].value_counts())
    PPF.filter_by_pet_age(84)
    PPF.drop_rows_by_key("Sterilized", "Not Sure")
    PPF.drop_rows_by_key("Health", "Minor injury")
    PPF.drop_rows_by_key("Health", "Serious injury")
    pos = PPF.df.copy()
    neg = PPF.df.copy()
    pos = pos[pos["AdoptionSpeed"] == 1]
    neg = neg[neg["AdoptionSpeed"] == 0]
    print(len(pos))
    pos = pos.sample(frac=0.40)
    print(len(pos))
    print(len(neg))
    df = pd.concat([pos, neg])  #* with frac=0.4 pos:neg as follows 2500:3000 
    print(df)
    assert 0

def test_visualize_data(PPF):
    PPF.visualize_data("Age", "Fee")
    assert 0

def test_drop_rows_by_key(PPF):
    print(PPF.df)
    PPF.drop_rows_by_key("Type", "Cat")
    print(PPF.df)
    assert 0


def test_visualize_both(PPF):
    df1 = PPF.df.copy()
    df2 = PPF.df.copy()
    df1 = df1[df1["AdoptionSpeed"] == 0]
    df2 = df2[df2["AdoptionSpeed"] == 1]
    
    counts_neg = df1["MaturitySize"].value_counts(sort=True)
    counts_pos = df2["MaturitySize"].value_counts(sort=True)
    labels = counts_neg.axes[0]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, counts_neg.values, width, label='Neg')
    rects2 = ax.bar(x + width/2, counts_pos.values, width, label='Pos')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.savefig("MaturitySize_both.png")
    assert 0

def test_visualize_both_norm(PPF):
    df1 = PPF.df.copy()
    df2 = PPF.df.copy()
    df1 = df1[df1["AdoptionSpeed"] == 0]
    df2 = df2[df2["AdoptionSpeed"] == 1]

    col_name = "FurLength"

    sm1 = df1[col_name].value_counts().sum()
    sm2 = df2[col_name].value_counts().sum()
    counts_neg = df1[col_name].value_counts(sort=True)
    counts_pos = df2[col_name].value_counts(sort=True)

    # counts_neg = counts_neg[counts_neg/sm1 > 0.05]
    # counts_pos = counts_pos[counts_pos/sm2 > 0.05]

    counts_neg = counts_neg/counts_neg.sum()
    counts_pos = counts_pos/counts_pos.sum()

    labels = counts_neg.axes[0]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, counts_neg.values, width, label='Neg')
    rects2 = ax.bar(x + width/2, counts_pos.values, width, label='Pos')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.savefig(f"{col_name}_both_norm.png")
    assert 0

def test_visualize_negative(PPF):
    df2 = PPF.df.copy()
    df2 = df2[df2["AdoptionSpeed"] == 0]
    # counts = df2["Age"].value_counts()
    # sm = df2["Age"].value_counts().sum()
    col_name = "PhotoAmt"
    counts = df2[col_name].value_counts()
    sm = df2[col_name].value_counts().sum()
    print(counts)
    counts = counts[counts/sm > 0.005]  #* takes to statistics only ages with at least 0.5% frequency to avoid outliers  
    sorted_counts = counts.sort_index()
    # print()
    fig, ax = plt.subplots()
    x = np.arange(len(sorted_counts.axes[0]))
    ax.bar(x, sorted_counts.values)
    ax.set_xticks(x, sorted_counts.axes[0]) 
    plt.savefig(f"{col_name}_0.5%_neg.png")
    # df1 = df1[df1["Age"].value_coun
    assert 0


def test_visualize_positive(PPF):
    df1 = PPF.df.copy()
    df1 = df1[df1["AdoptionSpeed"] == 1]
    # counts = df1["Age"].value_counts()
    # sm = df1["Age"].value_counts().sum()
    col_name = "PhotoAmt"
    counts = df1[col_name].value_counts()
    sm = df1[col_name].value_counts().sum()
    print(counts)
    counts = counts[counts/sm > 0.005]  #* takes to statistics only ages with at least 0.5% frequency to avoid outliers  
    sorted_counts = counts.sort_index()
    # print()
    fig, ax = plt.subplots()
    x = np.arange(len(sorted_counts.axes[0]))
    ax.bar(x, sorted_counts.values)
    ax.set_xticks(x, sorted_counts.axes[0]) 
    plt.savefig(f"{col_name}_0.5%_pos.png")
    # df1 = df1[df1["Age"].value_counts().values/sm > 0.05]
    # print(df1["Age"].count())
    # fbk["Age"] = fbk["Age"]/fbk["Age"].mean()
    # counts = df["Age"].value_counts(sort=True)
    # print(counts)
    # counts = df["Breed1"].value_counts(sort=True)
    # print(counts)
    # # print(counts.axes)
    # # print(counts.values)
    # plt.bar(counts.axes[0], counts.values)
    # print(plt.savefig("1234_pos.png"))
    assert 0