from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import wandb

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


if __name__ == "__main__":
    # Инициализируем WandB
    wandb.init(
        project="digit-classification",
        name="logistic-regression",
        config={
            "max_iter": config["logistic_regression"]["max_iter"],
            "random_state": config["random_state"],
            "model_type": "LogisticRegression"
        }
    )
    
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
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(coef)), coef)
            plt.title(f'Coefficients for Class {i}')
            plt.xlabel('Feature Index')
            plt.ylabel('Coefficient Value')
            wandb.log({f"coefficient_class_{i}": wandb.Image(plt)})
            plt.close()
    
    # Логируем информацию о регуляризации
    wandb.log({"regularization_C": logistic_regression_model.C})
    
    test(logistic_regression_model, data["x_test"], data["y_test"])
    
    # Завершаем сессию WandB
    wandb.finish()
    
    print("Эксперимент с логистической регрессией завершен и залогирован в Weights & Biases.") 