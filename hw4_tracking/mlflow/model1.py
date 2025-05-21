from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

from config import config
from data import get_data


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def log_confusion_matrix(cm, labels=None):
    """Создает и сохраняет визуализацию матрицы ошибок"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Устанавливаем метки, если они предоставлены
    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    # Добавляем аннотации с количеством в каждой ячейке
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.0f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    # Сохраняем матрицу ошибок как артефакт
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close(fig)
    
    return confusion_matrix_path


def test(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    
    # Рассчитываем метрики
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    # Для ROC AUC используем one-vs-rest для многоклассовой классификации
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    
    # Выводим метрики в консоль
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    
    # Создаем и сохраняем визуализацию матрицы ошибок
    cm_path = log_confusion_matrix(cm, labels=list(range(10)))
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix_path": cm_path
    }


if __name__ == "__main__":
    # Устанавливаем URI для MLFlow (по умолчанию http://localhost:5000)
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Создаем или устанавливаем эксперимент MLFlow
    mlflow.set_experiment("Digit Classification")
    
    # Запускаем новый run в MLFlow для отслеживания
    with mlflow.start_run(run_name="LogisticRegression"):
        # Фиксируем параметры модели
        mlflow.log_param("max_iter", config["logistic_regression"]["max_iter"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Создаем модель
        logistic_regression_model = LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
            random_state=config["random_state"],
        )
        
        data = get_data()
        train(logistic_regression_model, data["x_train"], data["y_train"])
        
        # Логируем коэффициенты регрессии
        if hasattr(logistic_regression_model, 'coef_'):
            for i, coef in enumerate(logistic_regression_model.coef_):
                coef_fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(coef)), coef)
                ax.set_title(f'Coefficients for Class {i}')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Coefficient Value')
                
                # Сохраняем визуализацию коэффициентов
                coef_path = f"coef_class_{i}.png"
                plt.savefig(coef_path)
                plt.close(coef_fig)
                
                # Логируем изображение в MLFlow
                mlflow.log_artifact(coef_path)
                
                # Удаляем временный файл
                os.remove(coef_path)
        
        # Логируем информацию о регуляризации
        mlflow.log_param("regularization_C", logistic_regression_model.C)
        
        # Оцениваем модель и логируем метрики
        metrics = test(logistic_regression_model, data["x_test"], data["y_test"])
        
        # Логируем метрики в MLFlow
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        
        # Логируем матрицу ошибок
        mlflow.log_artifact(metrics["confusion_matrix_path"])
        os.remove(metrics["confusion_matrix_path"])
        
        # Логируем саму модель
        mlflow.sklearn.log_model(logistic_regression_model, "model")
        
        print("Эксперимент с логистической регрессией завершен и залогирован в MLFlow.") 