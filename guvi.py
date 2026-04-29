
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

common_cols = train_df.columns.intersection(test_df.columns)
train_df = train_df[common_cols]
test_df = test_df[common_cols]

target_col = "prognosis"

X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)


imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(max_depth=5)  # limit depth
}

results = {}

for name, model in models.items():
    
    # Cross Validation (IMPORTANT)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average='macro')
    
    results[name] = {
        "model": model,
        "accuracy": acc,
        "recall": recall,
        "cv_mean": np.mean(cv_scores)
    }
    
    print(f"\n{name}")
    print(f"Cross-Val Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(classification_report(y_val, y_pred))


best_model_name = max(results, key=lambda x: results[x]["cv_mean"])
best_model = results[best_model_name]["model"]

print("\nBest Model:", best_model_name)


y_pred_best = best_model.predict(X_val)

final_accuracy = accuracy_score(y_val, y_pred_best)
final_recall = recall_score(y_val, y_pred_best, average='macro')

print("\nFinal Performance")
print(f"Final Accuracy: {final_accuracy:.4f}")
print(f"Final Recall: {final_recall:.4f}")


cm = confusion_matrix(y_val, y_pred_best)

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


X_test = test_df.drop(columns=[target_col], errors='ignore')

train_columns = train_df.drop(columns=[target_col]).columns
X_test = X_test.reindex(columns=train_columns, fill_value=0)

X_test = imputer.transform(X_test)

test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)

test_preds_labels = le.inverse_transform(test_preds)


output_df = pd.DataFrame({
    "Predicted Disease": test_preds_labels,
    "Confidence Score": np.max(test_probs, axis=1)
})

print("\nSample Predictions:")
print(output_df.head())
