from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, X_test, y_test, model_name, threshold=None):
    if threshold is not None:
        y_prob = model.predict(X_test)
        y_pred = (y_prob > threshold).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(matrix)
    print("-" * 30)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ConfusionMatrix": matrix
    }