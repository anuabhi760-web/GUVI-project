
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report


train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")




if 'Unnamed: 133' in train.columns:
    train = train.drop(columns=['Unnamed: 133'])


train = train.fillna(0)
test = test.fillna(0)


le = LabelEncoder()
train['prognosis'] = le.fit_transform(train['prognosis'])
test['prognosis'] = le.transform(test['prognosis'])


X_train = train.drop('prognosis', axis=1)
y_train = train['prognosis']

X_test = test.drop('prognosis', axis=1)
y_test = test['prognosis']



model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,        # restrict depth to avoid overfitting
    random_state=42
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)



accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



importances = model.feature_importances_
features = X_train.columns

feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Symptoms:\n")
print(feature_df.head(10))


def predict_disease(input_symptoms):
    input_df = pd.DataFrame([input_symptoms], columns=X_train.columns)
    
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    disease = le.inverse_transform(prediction)[0]
    
    prob_df = pd.DataFrame(probabilities, columns=le.classes_)
    prob_df = prob_df.T.sort_values(by=0, ascending=False)
    
    print("\nPredicted Disease:", disease)
    print("\nTop Probabilities:\n", prob_df.head())

sample = [0] * X_train.shape[1]
sample[0] = 1  

predict_disease(sample)
