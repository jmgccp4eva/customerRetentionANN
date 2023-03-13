import numpy as np
import pandas as pd
import tensorflow as tf     #CURRENT VERSION ON 3/13/23 IS 2.11.0
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # NOTE sklearn installed through scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# DATA PREPROCESSING
    # IMPORT DATASET
dataset = pd.read_csv("Churn_Modelling.csv")
MATRIX_X = dataset.iloc[:, 3:-1].values
MATRIX_Y = dataset.iloc[:, -1].values

    # IS THERE ANY MISSING DATA?    NO

    # ENCODING CATEGORICAL DATA
        # GENDER CONVERT (FEMALE=0 AND MALE=1)
le = LabelEncoder()
MATRIX_X[:, 2] = le.fit_transform(MATRIX_X[:, 2])

        # ONE HOT ENCODING OF GEORGRAPHY COLUMN
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
MATRIX_X = np.array(ct.fit_transform(MATRIX_X))

    # SPLIT DATASET INTO TRAINING AND TEST SETS
X_train, X_test, Y_train, Y_test = train_test_split(MATRIX_X, MATRIX_Y, test_size=0.2, random_state=0)

    # FEATURE SCALING!!!!
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# BUILDING THE ANN
ann = tf.keras.models.Sequential()      # INITIALIZATION
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # CREATES SHALLOW NN OF 6 UNITS USING RELU --FIRST LAYER ONLY
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # ADDING A 2ND LAYER
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # ADDING A BINARY OUTPUT LAYER

# TRAINING THE ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)  # 86% ACCURACY

# MAKING PREDICTIONS
prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
prediction = prediction[0][0]*100
prediction = f"There is a {prediction}% chance that this customer will leave the bank!"
print(prediction)

# PREDICTING TEST SET RESULTS
Y_prediction = ann.predict(X_test)
Y_prediction = (Y_prediction > 0.5)
print(np.concatenate((Y_prediction.reshape(len(Y_prediction), 1), Y_test.reshape(len(Y_test), 1)), 1))

# MAKE CONFUSION MATRIX
cm = confusion_matrix(Y_test, Y_prediction)
print(cm)
ac_sc = accuracy_score(Y_test, Y_prediction)
ac_sc = ac_sc*100
ac_sc = f"Accuracy of {ac_sc}%"
print(ac_sc)