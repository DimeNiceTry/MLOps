from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
import tempfile

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


def log_tree_visualization(model, feature_names=None, class_names=None):
    """Создает и сохраняет визуализацию дерева решений"""
    # Ограничиваем глубину для удобства визуализации
    max_depth = min(3, model.max_depth) if model.max_depth else 3
    
    # Создаем tempfile для хранения DOT-данных
    dot_data_file = tempfile.NamedTemporaryFile(suffix='.dot', delete=False)
    dot_file_path = dot_data_file.name
    dot_data_file.close()
    
    # Экспортируем дерево в DOT-формат
    export_graphviz(
        model,
        out_file=dot_file_path,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )
    
    return dot_file_path


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
    with mlflow.start_run(run_name="DecisionTree"):
        # Фиксируем параметры модели
        mlflow.log_param("max_depth", config["decision_tree"]["max_depth"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("model_type", "DecisionTree")
        
        # Создаем модель
        decision_tree_model = DecisionTreeClassifier(
            random_state=config["random_state"],
            max_depth=config["decision_tree"]["max_depth"]
        )
        
        data = get_data()
        train(decision_tree_model, data["x_train"], data["y_train"])
        
        # Логируем параметры дерева
        mlflow.log_param("criterion", decision_tree_model.criterion)
        mlflow.log_metric("node_count", decision_tree_model.tree_.node_count)
        mlflow.log_metric("leaf_count", decision_tree_model.tree_.n_leaves)
        
        # Логируем важность признаков
        feature_importance = decision_tree_model.feature_importances_
        
        # Создаем и сохраняем график важности признаков
        fig, ax = plt.subplots(figsize=(12, 6))
        feature_idx = np.arange(len(feature_importance))
        ax.bar(feature_idx, feature_importance)
        ax.set_title('Feature Importance')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Importance')
        
        importance_path = "feature_importance.png"
        plt.savefig(importance_path)
        plt.close(fig)
        
        # Логируем изображение в MLFlow
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)
        
        # Логируем важность каждого признака отдельно
        for i, importance in enumerate(feature_importance):
            mlflow.log_metric(f"feature_importance_{i}", importance)
        
        # Пробуем сохранить визуализацию дерева
        try:
            tree_dot_path = log_tree_visualization(decision_tree_model)
            mlflow.log_artifact(tree_dot_path, "tree_visualization")
            os.remove(tree_dot_path)
        except Exception as e:
            print(f"Не удалось создать визуализацию дерева: {e}")
        
        # Оцениваем модель и логируем метрики
        metrics = test(decision_tree_model, data["x_test"], data["y_test"])
        
        # Логируем метрики в MLFlow
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        
        # Логируем матрицу ошибок
        mlflow.log_artifact(metrics["confusion_matrix_path"])
        os.remove(metrics["confusion_matrix_path"])
        
        # Логируем саму модель
        mlflow.sklearn.log_model(decision_tree_model, "model")
        
        print("Эксперимент с решающим деревом завершен и залогирован в MLFlow.") 