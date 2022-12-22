from re import M
import pytest
from src.preprocessing import load_csv, PreprocessPetFinder
import numpy as np

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


def test_ppf_general(PPF):
    print(PPF.df.head())
    print(PPF.df.columns)
    print(PPF.df[["Color1", "Color2", "MaturitySize", "FurLength"]])
    assert 0

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
    encoding_layer = PPF.encode_categorical(train_ds, "Type", "string")
    for v, _ in train_ds:
        print(encoding_layer(v["Type"]))
    # print(encodeding_layer(train_feature))
    assert 0

def test_get_imbalance(PPF):
    res = PPF.df["AdoptionSpeed"].value_counts()
    print(res[0]/res[1])
    assert 0

def test_visualize_data(PPF):
    PPF.visualize_data("Age", "Fee")
    assert 0

def test_drop_rows_by_key(PPF):
    print(PPF.df)
    PPF.drop_rows_by_key("Type", "Cat")
    print(PPF.df)
    assert 0