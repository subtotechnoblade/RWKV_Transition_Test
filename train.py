import tensorflow as tf
from model import make_toy_model
from Functions import make_train_dataset, make_test_dataset
from Acc import SCCA_general
def train():
    model = make_toy_model(128, 2, 3, lamb=5.0, alpha=0.98)

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3,
                                                      weight_decay=1e-3,
                                                      beta_1=0.98),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1,),
                  metrics=[SCCA_general])

    x_train, y_train = make_train_dataset(16 * 1000, 75, use_noise=True)
    # x_train, y_train = make_test_dataset(16 * 1000, 50, use_noise=False)
    x_test, y_test = make_test_dataset(16 * 100, 100)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_test, y_test),
              shuffle=True,
              batch_size=16,
              epochs=20)
    model.save_weights("model.weights.h5")

if __name__ == "__main__":
    train()
