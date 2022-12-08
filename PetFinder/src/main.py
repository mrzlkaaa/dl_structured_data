import tensorflow as tf


class Model:
    def __init__(self):
        return

    def model_setup_sparse(self, **layers):
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(256)(features)
        # x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        x = tf.keras.layers.Dense(5)(x)
        model = tf.keras.Model(input_layers, x)
        model.summary()
        return model

    def model_setup_binary(self, **layers):
        encoded_layers = layers.get('encoded_layers')
        input_layers = layers.get('input_layers')
        features = tf.keras.layers.concatenate(encoded_layers)
        x = tf.keras.layers.Dense(32)(features)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(input_layers, x)
        model.summary()
        return model

    def model_compiler(self, model):
        model.compile(optimizer="adam",
        #? pick the loss based on task via switch
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
        )
        return model
    
    