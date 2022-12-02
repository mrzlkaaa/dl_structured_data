import tensorflow as tf
import pandas as pd
import os

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
file_name = "petfinder-mini.csv"

def load_csv(url):
    path = tf.keras.utils.get_file("petfinder_mini.zip", url, extract=True, cache_subdir="")
    file_path = os.path.join(os.path.split(path)[0], "datasets/petfinder-mini", file_name)
    csv = pd.read_csv(file_path)
    return csv

class PreprocessPetFinder:
    def __init__(self, df, target): #* target = AdoptionSpeed
        self.df = df
        self.target = target
        self.df_train, self.df_val, self.df_train

    def drop_target_col(self, df):
        df_featured = df.copy()
        labels = df_featured.pop(self.target)
        return df_featured, labels