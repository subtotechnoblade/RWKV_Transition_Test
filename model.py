import tensorflow as tf
from RWKV_v6 import RWKV_Block
from Batch_Wrapper import Batch
from Grok_Model import Grok_Fast_EMA_Model
import matplotlib.pyplot as plt

# 1
def make_model(embed_size, num_head=1, num_layers=2):
    inputs = tf.keras.layers.Input(batch_shape=(None, None, 50), name="inputs")

    x = Batch(tf.keras.layers.Dense(embed_size))(inputs)

    for layer_id in range(num_layers):
        x = RWKV_Block(layer_id, num_head, embed_size)(x)

    x = Batch(tf.keras.layers.Dense(6))(x)

    outputs = tf.keras.layers.Activation("softmax")(x)
    return Grok_Fast_EMA_Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    model = make_model(128, 1, 2)
    model.summary()






