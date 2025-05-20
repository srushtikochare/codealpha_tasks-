

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/Users/srushtikochare/Downloads/large_disease_dataset.csv")  # Replace with your actual file name


print("Shape of data:", df.shape)
print("Sample data:\n", df.head())


df.dropna(inplace=True)


X = df.drop("Disease", axis=1)
y = df["Disease"]


X = pd.get_dummies(X)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns


plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10])
plt.title("Top 10 Important Features (Symptoms)")
plt.xlabel("Importance")
plt.ylabel("Symptoms")
plt.tight_layout()
plt.show()
