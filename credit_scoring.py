import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

np.random.seed(42)  

n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.randint(21, 65, size=n_samples),
    'Income': np.random.randint(20000, 120000, size=n_samples),
    'LoanAmount': np.random.randint(1000, 50000, size=n_samples),
    'CreditHistory': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8])
})


data['Creditworthy'] = (
    (data['Income'] > 40000).astype(int) +
    (data['CreditHistory'] == 1).astype(int) +
    (data['Age'] > 30).astype(int) +
    (data['LoanAmount'] < 30000).astype(int)
)


data['Creditworthy'] = (data['Creditworthy'] >= 3).astype(int)

print("Sample of dataset:")
print(data.head())



X = data.drop('Creditworthy', axis=1)
y = data['Creditworthy']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LogisticRegression()
model.fit(X_train_scaled, y_train)



y_pred = model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
