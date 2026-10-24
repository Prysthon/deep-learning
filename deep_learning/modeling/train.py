import keras
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)


def build_model():
    model = keras.Sequential()

    # Camada de entrada
    model.add(Input(shape=(32, 32, 3)))

    # Bloco 1
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloco 2
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloco 3
    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Transição CNN -> Densa
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))

    # Saída
    model.add(Dense(10, activation="softmax"))

    # Compilar
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model


def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.2)

    return history
