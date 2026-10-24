from deep_learning.config import class_names
from deep_learning.dataset import load_dataset
from deep_learning.modeling.predict import evaluate_model, predict
from deep_learning.modeling.train import build_model, train_model
from deep_learning.plots import (
    plot_confusion_matrix,
    plot_examples,
    plot_history,
    plot_predictions,
)


def main():
    X_train, X_test, y_train, y_test = load_dataset()
    plot_examples(X_train, y_train, class_names)
    model = build_model()
    history = train_model(model, X_train, y_train)
    plot_history(history)
    evaluate_model(model, X_test, y_test)
    y_pred = predict(model, X_test)
    plot_predictions(X_test, y_train, y_test, y_pred, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()
