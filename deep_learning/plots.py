import random

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_examples(X_train, y_train, class_names):
    plt.figure(figsize=(10, 10))

    for i in range(9):
        num = random.randint(1, 2000)
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[num])
        plt.title(class_names[y_train[num]])

        plt.axis('off')

    plt.suptitle('Exemplos da base CIFAR-10.')

    plt.tight_layout()

    plt.show()


def plot_history(history):
    plt.figure(figsize=(8, 5))

    plt.plot(history.history['accuracy'], label="Train")

    plt.plot(history.history['val_accuracy'], label="validation")

    plt.title('acuracia por epoch')

    plt.xlabel('epoch')
    plt.ylabel('acuracia')

    plt.legend()
    plt.show()
    ######
    plt.figure(figsize=(8, 5))

    plt.plot(history.history['loss'], label="Train")

    plt.plot(history.history['val_loss'], label="validation")

    plt.title('loss por epoch')

    plt.xlabel('epoch')
    plt.ylabel('acuracia')

    plt.legend()
    plt.show()


def plot_predictions(X_test, y_train, y_test, y_pred, class_names):
    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[i])
        plt.title(class_names[y_train[i]])
        real = class_names[y_test[i]]
        pred = class_names[y_pred[i]]

        plt.title(f"Real: {real}\nPredição: {pred}")
    
        plt.axis('off')

    plt.suptitle('predições na base de Teste CIFAR-10.')

    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title("Matriz de Confusão")
    plt.show()
