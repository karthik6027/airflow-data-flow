# model_evaluation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load Preprocessed Data
# ---------------------------

def model_prediction():
    df = pd.read_csv('data\preprocessed_data\cleaned_customer_churn.csv')

    # ---------------------------
    # 2. Define Features and Target Variable
    # ---------------------------
    # Assuming 'Churn' is the target variable.
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # ---------------------------
    # 3. Split the Data
    # ---------------------------
    # Use a 70:30 train-test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # ---------------------------
    # 4. Random Forest Model
    # ---------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    print("=== Random Forest Model Evaluation ===")
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print("Accuracy:", rf_accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))
    print("Classification Report:")
    print(classification_report(y_test, rf_predictions))

    # Visualize confusion matrix for Random Forest
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    plt.savefig('rf_confusion_matrix.png')
    plt.close()

    # ---------------------------
    # 5. Logistic Regression Model
    # ---------------------------
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    print("\n=== Logistic Regression Model Evaluation ===")
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print("Accuracy:", lr_accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_predictions))
    print("Classification Report:")
    print(classification_report(y_test, lr_predictions))

    # Visualize confusion matrix for Logistic Regression
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, lr_predictions), annot=True, fmt='d', cmap='Greens')
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig('lr_confusion_matrix.png')
    plt.close()

    # ---------------------------
    # 6. Additional Model Evaluation Metrics
    # ---------------------------
    # If needed, one could also add ROC curve plotting or other metrics here as in the notebook.
    from sklearn.metrics import roc_curve, auc

    # For Random Forest
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(6,4))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc='lower right')
    plt.savefig('rf_roc_curve.png')
    plt.close()

    # For Logistic Regression
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure(figsize=(6,4))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc='lower right')
    plt.savefig('lr_roc_curve.png')
    plt.close()

    print("Model evaluation completed. All plots have been saved.")
    return "Model evaluation completed. All plots have been saved."
