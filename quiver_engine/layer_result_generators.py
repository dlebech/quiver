import tensorflow as tf


def get_outputs_generator(model, layer_name):
    return tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    ).predict
