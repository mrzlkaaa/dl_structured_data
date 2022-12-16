from preprocessing import PreprocessPetFinder, load_csv
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from vizualize_fit.main import FitVisualizer

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
tf.random.set_seed(42)

class Model:
    def __init__(self):
        return

    #? apply callback and find best lr for both binary and multiclass classifications
    def callback_lr(self):
        return LearningRateScheduler(lambda x: 0.001*10**(x/20))
        

    def model_setup_sparse(self, **layers): #* 84% accuracy with 50 epochs on test ds a quick setup (still can be simply improved)
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(512, activation="relu")(features)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        # x = tf.keras.layers.Dense(16)(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(5, activation="softmax")(x)
        model = tf.keras.Model(input_layers, x)
        model.summary()
        return model

    def model_setup_binary(self, **layers): #* 94% accuracy with 50 epochs on test ds (still can be simply improved)
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(32, activation="relu")(features)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        # x = tf.keras.layers.Dense(64, activation="relu")(x)
        # x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(input_layers, x)
        model.summary()
        return model

    def model_compiler_sparse(self, model):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      # ? pick the loss based on task via switch
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=False),
                      # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["accuracy"]
                      )
        return model

    def model_compiler_binary(self, model):
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      # ? pick the loss based on task via switch
                    #   loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    #       from_logits=False),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=["accuracy"]
                      )
        return model

if __name__ == "__main__":
    csv = load_csv(url)
    PPF = PreprocessPetFinder(csv, "AdoptionSpeed", "Description")
    train_df, val_df, test_df = PPF.split_df()
    train = PPF.df_to_ds(train_df)
    val = PPF.df_to_ds(val_df)
    test = PPF.df_to_ds(test_df)
    train_inp, train_decoded = PPF.ds_encode(train)

    M = Model()
    # model = M.model_setup_sparse(
    #     input_layers=train_inp, encoded_layers=train_decoded)
    # model = M.model_compiler_sparse(model)
    model = M.model_setup_binary(
        input_layers=train_inp, encoded_layers=train_decoded)
    model = M.model_compiler_binary(model)
    epochs = 50
    history = model.fit(train, epochs=epochs, validation_data=val)
    #* save data as a plot
    fv = FitVisualizer()
    fv.plot(history.history, epochs)
    fv.savefig()


