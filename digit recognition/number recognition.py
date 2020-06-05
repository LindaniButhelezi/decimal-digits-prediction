"""
Learning to predict decimal digits represented by an array of numbers
by L Buthelezi- BScEng(Chemical Engineering)
Location: Durban, South Africa
Email: L.Buthelezi@alumni.uct.ac.za
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns

dataset = pd.read_csv('pendigits.csv')
X = dataset.iloc[:, 0:-2].values 
y = dataset.iloc[:, -1].values 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train[:,[1,2]], y_train

sns.heatmap(cm, annot=True,linewidths=0.5,square=True, cmap='plasma')
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.show()

