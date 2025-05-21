from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image
import pydot

from config import config
from data import get_data


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    
    # Рассчитываем метрики
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    # Для ROC AUC используем one-vs-rest для многоклассовой классификации
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    
    # Логируем метрики в WandB
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    
    wandb.log({"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc})
    
    # Создаем матрицу ошибок для wandb
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test,
        preds=y_pred,
        class_names=list(range(10))
    )})
    
    # Создаем кривые ROC для каждого класса
    for i in range(10):  # 10 классов в датасете digits
        fpr, tpr, _ = roc_curve(
            (y_test == i).astype(int),
            y_proba[:, i]
        )
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.title(f'ROC Curve for Class {i}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        wandb.log({f"roc_curve_class_{i}": wandb.Image(plt)})
        plt.close()


def log_tree_visualization(model, feature_names=None, class_names=None):
    # Создаем DOT-представление дерева решений
    dot_data = io.StringIO()
    export_graphviz(
        model,
        out_file=dot_data,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3  # Ограничиваем глубину для читаемости
    )
    
    # Преобразуем DOT в изображение
    try:
        graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
        image_data = io.BytesIO(graph.create_png())
        image = Image.open(image_data)
        
        # Логируем изображение дерева в WandB
        wandb.log({"decision_tree_visualization": wandb.Image(image)})
    except Exception as e:
        print(f"Ошибка при визуализации дерева: {e}")


if __name__ == "__main__":
    # Инициализируем WandB
    wandb.init(
        project="digit-classification",
        name="decision-tree",
        config={
            "max_depth": config["decision_tree"]["max_depth"],
            "random_state": config["random_state"],
            "model_type": "DecisionTree"
        }
    )
    
    # Создаем модель
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )
    
    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    
    # Логируем параметры дерева
    wandb.log({
        "tree_depth": decision_tree_model.max_depth,
        "node_count": decision_tree_model.tree_.node_count,
        "leaf_count": decision_tree_model.tree_.n_leaves,
        "criterion": decision_tree_model.criterion
    })
    
    # Логируем важность признаков
    feature_importance = decision_tree_model.feature_importances_
    feature_idx = np.arange(len(feature_importance))
    
    plt.figure(figsize=(12, 6))
    plt.bar(feature_idx, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    wandb.log({"feature_importance": wandb.Image(plt)})
    plt.close()
    
    # Создаем таблицу важности признаков
    feature_importance_table = wandb.Table(
        columns=["Feature", "Importance"],
        data=[[f"Feature {i}", importance] for i, importance in enumerate(feature_importance)]
    )
    wandb.log({"feature_importance_table": feature_importance_table})
    
    # Визуализируем дерево
    try:
        log_tree_visualization(decision_tree_model)
    except ImportError:
        print("Пакет pydot не установлен, визуализация дерева пропущена")
    
    test(decision_tree_model, data["x_test"], data["y_test"])
    
    # Завершаем сессию WandB
    wandb.finish()
    
    print("Эксперимент с решающим деревом завершен и залогирован в Weights & Biases.") 