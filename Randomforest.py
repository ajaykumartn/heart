import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data)
X = data.drop("target", axis=1) 
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

model =  RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")