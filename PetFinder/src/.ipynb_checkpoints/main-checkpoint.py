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

    METRICS = [
        # tf.keras.metrics.TruePositives(name='tp'),
        # tf.keras.metrics.FalsePositives(name='fp'),
        # tf.keras.metrics.TrueNegatives(name='tn'),
        # tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        # tf.keras.metrics.AUC(name='auc'),
        # tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    def __init__(self):
        return

    # ? apply callback and find best lr for both binary and multiclass classifications
    def callback_lr(self):
        return LearningRateScheduler(lambda x: 0.001*10**(x/20))

    #! sparse
    # * 84% accuracy with 50 epochs on train ds a quick setup (overfitted)
    def model_setup_sparse(self, **layers):
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(16, activation="relu")(features)
        x = tf.keras.layers.Dense(5, activation="softmax")(x)
        model = tf.keras.Model(input_layers, x)
        # model.summary()
        return model

    #! binary
    # * 94% accuracy with 50 epochs on train ds (overfitted)
    def model_setup_binary(self, **layers):
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(16, activation="relu")(features)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(input_layers, x)
        return model

    def model_compiler_sparse(self, model):
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      # ? pick the loss based on task via switch
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"]
                      )
        return model

    def model_compiler_binary(self, model):
        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      # ? pick the loss based on task via switch
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=self.METRICS
                      )
        return model


if __name__ == "__main__":
    csv = load_csv(url)
    # * ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
    #*    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt',
    # *    'AdoptionSpeed']
    # * only "Description columns was dropped"
    # PPF = PreprocessPetFinder(csv, "AdoptionSpeed", "Description")
    # train_df, val_df, test_df = PPF.split_df()
    # train = PPF.df_to_ds(train_df)
    # val = PPF.df_to_ds(val_df)
    # test = PPF.df_to_ds(test_df)
    # train_inp, train_decoded = PPF.ds_encode(train)

    # * To simplify input dataset few more columns were dropped
    PPF = PreprocessPetFinder(csv, "AdoptionSpeed", "Description", "Fee", "PhotoAmt") #, "Breed1", "MaturitySize", "Color1", "Color2", "FurLength")
    PPF.binning_feature("Age")
    # PPF.drop_rows_by_key("Type", "Cat") #*  dropped one pet type -  only dogs left
    # PPF.filter_by_pet_age(84)
    # PPF.drop_rows_by_key("Sterilized", "Not Sure")
    # PPF.drop_rows_by_key("Health", "Minor injury")
    # PPF.drop_rows_by_key("Health", "Serious injury")

    # df_s = PPF.manual_undersampling(0.4)
    # df_s = PPF.manual_oversampling()
    # PPF.df = df_s
    # PPF.drop_rows_by_key("Vaccinated", "Not Sure")
    
    train_df, val_df, test_df = PPF.split_df()
    print(train_df)
    train = PPF.df_to_ds(train_df)
    val = PPF.df_to_ds(val_df)
    test = PPF.df_to_ds(test_df)
    train_inp, train_decoded = PPF.ds_encode(train)
    M = Model()
    model = M.model_setup_sparse(
        input_layers=train_inp, encoded_layers=train_decoded)
    model = M.model_compiler_sparse(model)

    print(train, val)
    # model = M.model_setup_binary(
    #     input_layers=train_inp, encoded_layers=train_decoded)
    # model = M.model_compiler_binary(model)

    epochs = 200
    history = model.fit(train, epochs=epochs, validation_data=val)
    print(history.history)
    # * save data as a plot
    fv = FitVisualizer(history.history)
    fv.plot(epochs)
    model.summary(print_fn=fv.save_model_summary)
    fv.save_fit_process()
    fv.savefig()
    model.evaluate(test)

# * On a given dataset the best results are 76% accuracy, val_accuracy 75% (tiny and small models)
# * The same results where achieved by varying the regularization (L1, L2) and by dropout layers
# * Continuous increasing of model with regularization, dropouts do not provide resuts improvenment and
# * on the other hand overfitting evidences are observed (incresing of validation loss, decrasting of validation accuracy when
# * training accuracy are continuously increasing )
# * To reach high accuracy on a given data there are few thoughts: drop some colums to simplify dataset (upd: does not lead to any improvenments)
# * increase of number of unique rows (increse the dataset in other words)
