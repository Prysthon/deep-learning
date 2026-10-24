import keras


def load_dataset():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # normalizando os valores para 0 e 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # transformar em um vetor
    y_train = y_train.flatten()
    y_test = y_test.flatten()


    print(f'Formato de X_train: {X_train.shape}')
    print(f'Formato de X_test: {X_test.shape}')
    print(f'Formato de y_train: {y_train.shape}')
    print(f'Formato de y_test: {y_test.shape}')

    return X_train, X_test, y_train, y_test
