from keras import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling2D


def build_discriminator():
    dis_model = Sequential()
    dis_model.add(Conv2D(128, (5, 5), padding="same", input_shape=(64, 64, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(LeakyReLU(alpha=0.2))

    dis_model.add(Dense(1))
    dis_model.add(Activation("sigmoid"))

    return dis_model
