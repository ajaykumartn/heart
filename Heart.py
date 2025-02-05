import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data)
X = data.drop("target", axis=1) 
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data)
X = data.drop("target", axis=1) 
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data)
X = data.drop("target", axis=1) 
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

model =  SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data)
X = data.drop("target", axis=1) 
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.29, random_state=42)

model =  KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")



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


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data)
X = data.drop("target", axis=1) 
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

model =  LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
