import numpy as np
from numba import njit
import matplotlib.pyplot as plt


# returns 1 segment which is a vector 100 of amplitudes
@njit(cache=True)
def Normal(length=50, noise=True):
    output = np.zeros(shape=(100 * length,), dtype=np.float64)
    if noise:
        output += np.random.uniform(low=-0.02, high=0.02, size=(100 * length,))
    return output.reshape((length, 100))

@njit(cache=True)
def Sin(length=50, noise=True):
    offset = np.random.randint(-100, 100)
    time = np.arange(100 * length) + offset
    output = np.sin(.1 * time)
    if noise:
        output += np.random.uniform(low=-0.02, high=0.02, size=(100 * length,))
    return output.reshape((length, 100))

@njit(cache=True)
def Abs_Sin(length=50, noise=True):
    offset = np.random.randint(-100, 100)
    time = np.arange(100 * length) + offset

    output = np.abs(np.sin(.1 * time))
    if noise:
        output += np.random.uniform(low=-0.02, high=0.02, size=(100 * length,))
    return output.reshape((length, 100))

@njit(cache=True)
def Triangle(length=50, noise=True):
    offset = np.random.randint(-100, 100)
    time = np.arange(100 * length) + offset
    output = (2 / np.pi) * np.arcsin(np.sin(.1 * time))
    if noise:
        output += np.random.uniform(low=-0.02, high=0.02, size=(100 * length,))
    return output.reshape((length, 100))

@njit(cache=True)
def Square(length=50, b=0.01, noise=True):
    offset = np.random.randint(-100, 100)
    time = (np.arange(100 * length) / 10) + offset
    sin_time = np.sin(time)
    output = sin_time / (((b ** 2) + (sin_time ** 2)) ** 0.5)
    if noise:
        output += np.random.uniform(low=-0.02, high=0.02, size=(100 * length,))
    return output.reshape((length, 100))

@njit(cache=True)
def Saw_Tooth(length=50, period=30, noise=True):
    offset = np.random.randint(-100, 100)
    time = np.arange(100 * length) + offset
    output = 2 * ((time % period) / period) - 1
    if noise:
        output += np.random.uniform(low=-0.02, high=0.02, size=(100 * length,))

    return output.reshape((length, 100))


def make_train_dataset(samples=16 * 100, time_steps=50, use_noise=True):
    # Normal : 40%
    # 0: Normal

    # Problems with 60%, thus each with 12% chance
    # 1: sin
    # 2: abs_sin
    # 3: triangle
    # 4: square
    # 5, Saw_tooth

    # amount of timesteps per batch is 50, each timestep having 100 segements
    # Note that every sequence will be the same type of wave
    # for 1000 samples
    x_train = [None] * samples
    y_train = [None] * samples
    funcs = [Normal, Sin, Abs_Sin, Triangle, Square, Saw_Tooth]
    for i in range(samples):
        chosen_type = np.random.choice(np.arange(0, 6), size=(1,), p=[.4, .12, .12, .12, .12, .12])[0]

        label_template = np.ones((100,), dtype=np.float32) * -1
        x_train[i] = funcs[chosen_type](length=time_steps, noise=use_noise)

        label_template[-1] = chosen_type
        label = np.repeat(np.expand_dims(label_template, 0), time_steps, 0)
        y_train[i] = label
    return np.array(x_train), np.array(y_train)

def make_test_dataset(samples=16 * 100, use_noise=True):
    x_test, y_test = [None] * samples, [None] * samples
    funcs = [Normal, Sin, Abs_Sin, Triangle, Square, Saw_Tooth]

    for i in range(samples):
        data = [None] * 50
        label = [None] * 50
        label_template = np.ones((100,), dtype=np.float32) * -1
        for j in range(50):
            chosen_type = np.random.choice(np.arange(0, 6), size=(1,), p=[.4, .12, .12, .12, .12, .12])[0]

            data[j] = funcs[chosen_type](length=1, noise=use_noise)[0]

            label_template[-1] = chosen_type
            label[j] = label_template

        x_test[i] = data
        y_test[i] = label
    return np.array(x_test), np.array(y_test)



    # return 0
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plt.plot(np.concatenate(Triangle(length=2)))
    # # plt.plot(Saw_Tooth()[1])
    # # plt.plot(Saw_Tooth()[2])
    # # plt.plot(Sin())
    # plt.ylabel('some numbers')
    # plt.show()
    # x_train, y_train = make_train_dataset()
    # print(x_train.shape)
    # print(y_train.shape)
    # make_test_dataset_func(Normal, Sin, Abs_Sin, Triangle, Square, Saw_Tooth)
    x_test, y_test = make_test_dataset(16 * 2, True)
    print(x_test.shape)
    print(y_test.shape)
    # print(Normal())
    # print(test(Normal, Sin, Abs_Sin, Triangle, Square, Saw_Tooth))
