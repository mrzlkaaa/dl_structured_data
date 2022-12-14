from preprocessing import PreprocessPetFinder, load_csv
import tensorflow as tf

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"


class Model:
    def __init__(self):
        return

    def model_setup_sparse(self, **layers):
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(32, activation="elu")(features)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation="elu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # x = tf.keras.layers.Dense(16)(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(5)(x)
        model = tf.keras.Model(input_layers, x)
        model.summary()
        return model

    def model_setup_binary(self, **layers):
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(32)(features)
        x = tf.keras.layers.Dense(32)(features)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(input_layers, x)
        model.summary()
        return model

    def model_compiler_sparse(self, model):
        model.compile(optimizer=tf.keras.optimizers.Adam(),
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
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["accuracy"]
                      )
        return model


if __name__ == "__main__":
    csv = load_csv(url)
    PPF = PreprocessPetFinder(csv, "AdoptionSpeed", "Description")
    PPF.increasing_df(0.7)
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
    model.fit(train, epochs=10, validation_data=val)
