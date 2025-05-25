from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task

from config import config
from data import get_data


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test, task) -> None:
    y_pred = model.predict(x_test)
    
    # Рассчитываем метрики
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    # Для ROC AUC нам нужны вероятности
    y_score = model.predict_proba(x_test)
    # Используем one-vs-rest для многоклассовой классификации
    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
    
    # Логируем метрики в ClearML
    print(f"Accuracy: {accuracy}")
    task.get_logger().report_scalar(title="Metrics", series="Accuracy", value=accuracy, iteration=0)
    print(f"F1 Score: {f1}")
    task.get_logger().report_scalar(title="Metrics", series="F1 Score", value=f1, iteration=0)
    print(f"ROC AUC: {roc_auc}")
    task.get_logger().report_scalar(title="Metrics", series="ROC AUC", value=roc_auc, iteration=0)
    
    # Логируем матрицу ошибок
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Сохраняем матрицу ошибок и логируем в ClearML
    plt.tight_layout()
    task.get_logger().report_matplotlib_figure(title="Confusion Matrix", series="", figure=fig)
    plt.close(fig)


if __name__ == "__main__":
    # Инициализируем задачу в ClearML
    task = Task.init(project_name="Digit Classification", task_name="DecisionTree", reuse_last_task_id=False)
    
    # Создаем модель
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )
    
    # Логируем параметры модели
    params = {
        "max_depth": config["decision_tree"]["max_depth"],
        "random_state": config["random_state"],
    }
    task.connect(params)
    
    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    
    # Логируем параметры обученного дерева
    task.get_logger().report_scalar(title="Tree Properties", series="Max Depth", value=decision_tree_model.max_depth, iteration=0)
    task.get_logger().report_scalar(title="Tree Properties", series="Node Count", value=decision_tree_model.tree_.node_count, iteration=0)
    task.get_logger().report_scalar(title="Tree Properties", series="Leaf Count", value=decision_tree_model.tree_.n_leaves, iteration=0)
    task.get_logger().report_text(f"Criterion: {decision_tree_model.criterion}")
    
    # Логируем важность признаков
    importance = decision_tree_model.feature_importances_
    task.get_logger().report_histogram(title="Feature Importance", series="Importance", values=importance, iteration=0)
    
    feature_ids = np.arange(len(importance))
    plt.figure(figsize=(10, 6))
    plt.barh(feature_ids, importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature ID')
    plt.title('Decision Tree Feature Importance')
    task.get_logger().report_matplotlib_figure(title="Feature Importance Plot", series="", figure=plt.gcf())
    plt.close()
    
    test(decision_tree_model, data["x_test"], data["y_test"], task)
    
    print("Эксперимент с решающим деревом завершен и залогирован в ClearML.") 