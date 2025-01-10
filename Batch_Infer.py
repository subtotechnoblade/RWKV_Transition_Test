import tensorflow as tf
class Batch(tf.keras.layers.Layer):
    def __init__(self, operation, **kwargs):
        super().__init__(**kwargs)
        self.operation = operation

    def call(self, inputs):
        return self.operation(inputs)