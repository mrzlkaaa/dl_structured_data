import tensorflow as tf
from tensorflow.keras.layers import StringLookup, IntegerLookup, Normalization, CategoryEncoding
import pandas as pd
import numpy as np
import os

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
file_name = "petfinder-mini.csv"


def load_csv(url):
    path = tf.keras.utils.get_file(
        "petfinder_mini.zip", url, extract=True, cache_subdir="")
    print(os.listdir(os.path.split(path)[0]))
    file_path = os.path.join(os.path.split(
        path)[0], "petfinder-mini", file_name)
    # file_path = os.path.join(os.path.split(path)[0], "datasets/petfinder-mini", file_name)
    csv = pd.read_csv(file_path)
    return csv


class PreprocessPetFinder:
    def __init__(self, df, target, *to_drop):  # * target = AdoptionSpeed
        self.df = self.drop_unused(df, to_drop)
        # self.train, self.val, self.test = self.split_df()
        self.target = target
        self.df_colsdata = self.get_df_colsdata()

    # def target_from_label(self, *target):
    #     return

    def drop_unused(self, df, to_drop):
        # df['AdoptionSpeed'] = np.where(df['AdoptionSpeed'] == 4, 0, 1) #* enabled only for binary classification
        new_df = df.drop(columns=list(to_drop))
        # print(new_df)
        new_df = new_df.dropna()
        return new_df

    #* calls before df splitting
    def increasing_df(self, frac):
        sampled_df = self.df.sample(frac=frac, random_state=123)
        self.df = pd.concat([self.df, sampled_df])


    def split_df(self):
        return np.split(self.df, [int(0.8*len(self.df)), int(0.9*len(self.df))])

    def format_dtype(self, dtype):
        match dtype:
            case "object":
                return "string"
            case "int64":
                return "int64"
            case _:
                raise ValueError

    def get_df_colsdata(self, format_dtype=True):
        # * populate the dictionary with info as follows: key - col_name, values - col_name, dtype
        # * avaliable optional formatting of dtype according on given data type in dataframe
        cols_data = dict()
        for i in range(len(self.df.columns)):
            cols_data[self.df.columns[i]] = {"name": self.df.columns[i],
                                             "dtype": self.format_dtype(self.df.dtypes[i]) if format_dtype else self.df.dtypes[i]}
        # print(cols_data)
        return cols_data

    def drop_target_col(self, df):
        df_featured = df.copy()
        try:
            labels = df_featured.pop(self.target)
            self.df_colsdata.pop(self.target)
        except KeyError:
            print("col is removed already")
        return df_featured, labels

    def df_to_ds(self, df):
        df_featured, labels = self.drop_target_col(df)
        extended_df = {key: value[:, tf.newaxis]
                       for key, value in df_featured.items()}
        ds = tf.data.Dataset.from_tensor_slices((extended_df, labels))
        ds = ds.shuffle(buffer_size=len(df_featured))
        ds = ds.batch(32)
        # print(ds)
        return ds

    def encode_numerical(self, ds, name):
        # * ds consists of numerical data and labels
        feature = ds.map(lambda x, y: x[name])
        norm = Normalization(axis=None)
        norm.adapt(feature)
        return norm

    def encode_categorical(self, ds, name, dtype, max_tokens=5):
        feature = ds.map(lambda x, y: x[name])
        if dtype == "string":
            indexes = StringLookup(max_tokens=max_tokens)
        elif dtype == "int64":
            indexes = IntegerLookup(max_tokens=max_tokens)
        indexes.adapt(feature)
        encoder = CategoryEncoding(
            num_tokens=indexes.vocabulary_size(),  output_mode="one_hot")
        return lambda f: encoder(indexes(f))

    def make_input_layer(self, name, dtype):
        return tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    def ds_encode(self, ds):
        # * consider 2 different data types: numerical, categorical
        # * make input layer and apply encoding on it
        numerical_colsdata = {key: value for key, value in self.df_colsdata.items()
                              if key in ["Age", "PhotoAmt", "Fee"]}
        categorical_colsdata = {key: value for key, value in self.df_colsdata.items()
                                if not key in ["Age", "PhotoAmt", "Fee", "AdoptionSpeed"]}
        input_layers = []
        encoded_layers = []

        for i in categorical_colsdata.values():
            input_layer = self.make_input_layer(i["name"], i["dtype"])
            # print(type(input_layer))
            encoding_layer = self.encode_categorical(ds, i["name"], i["dtype"])
            encoded_col = encoding_layer(input_layer)
            input_layers.append(input_layer)
            encoded_layers.append(encoded_col)

        for i in numerical_colsdata.values():
            input_layer = self.make_input_layer(i["name"], i["dtype"])
            encoding_layer = self.encode_numerical(ds, i["name"])
            encoded_col = encoding_layer(input_layer)
            input_layers.append(input_layer)
            encoded_layers.append(encoded_col)

        return input_layers, encoded_layers
