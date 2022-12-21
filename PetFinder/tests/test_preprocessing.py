from re import M
import pytest
from src.preprocessing import load_csv, PreprocessPetFinder
import numpy as np
from matplotlib import pyplot as plt

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"


@pytest.fixture
def csv():
    return load_csv(url)


@pytest.fixture
def PPF(csv):
    return PreprocessPetFinder(csv, "AdoptionSpeed", "Description")


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


# def test_filter_by_key(PPF):
#     df = PPF.filter_by_key(PPF.df, "AdoptionSpeed", 1)
#     assert 0


# def test_plot(fbk):
#     print(fbk["Age"])
#     # fbk["Age"] = fbk["Age"]/fbk["Age"].mean()
#     counts = fbk["Age"][fbk["Age"] < 50].value_counts(sort=True)
#     print(counts.axes)
#     print(counts.values)
#     plt.bar(counts.axes[0], counts.values)

#     # ax = counts.plot(kind="hist")
#     print(plt.savefig("hist.png"))
#     # ax.bar(counts[0], counts[1])
#     # plt.savefig("1234.png")
#     assert 0
