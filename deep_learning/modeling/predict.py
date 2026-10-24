import numpy as np


def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f'Acurácia: {test_accuracy * 100:.2f}%\nLoss: {test_loss:.3f}')

    return test_loss, test_accuracy


def predict(model, X_test):
    # Previções em imagens do Teste

    pred_proba = model.predict(X_test)
    y_pred = np.argmax(pred_proba, axis=1)

    return y_pred
