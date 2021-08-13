from sklearn.datasets import load_boston
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

knn = KNeighborsClassifier(n_neighbors=4)
rfc = RandomForestClassifier(n_estimators=300)
dtree = DecisionTreeClassifier

lm = LinearRegression()
boston = load_boston()


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
# print(df.head(5))

print(df.head())
sns.jointplot(x='CRIM', y='INDUS', data=df, kind='kde', color='red')
# sns.pairplot(df.drop(['ZN', 'CHAS', 'RAD'], axis=1))
plt.show()