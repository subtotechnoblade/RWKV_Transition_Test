import tensorflow as tf
from model import make_toy_model
from Functions import make_train_dataset, make_test_dataset
from Acc import SCCA_general
def train():
    model = make_toy_model(128, 2, 1)

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1,),
                  metrics=[SCCA_general])

    # x_train, y_train = make_train_dataset(16 * 100, 50)
    x_train, y_train = make_test_dataset(16 * 100, 50)
    x_test, y_test = make_test_dataset(16 * 10, 50)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_test, y_test),
              shuffle=True,
              batch_size=4,
              epochs=10)

if __name__ == "__main__":
    train()
