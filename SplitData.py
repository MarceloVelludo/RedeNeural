import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def splitData(caminho, testSize):

    iris = pd.read_csv(caminho, header=None)
    iris.columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

    # Criando uma classe numerica para as especies (0,1,2)
    iris.loc[iris[:]["class"] == 'Iris-setosa', 'class'] = 0
    iris.loc[iris[:]["class"] == 'Iris-versicolor', 'class'] = 1
    iris.loc[iris[:]["class"] == 'Iris-virginica', 'class'] = 2

    #Criando input e output
    x = iris[["sepal-length", "sepal-width", "petal-length", "petal-width"]].values.T
    y = iris[['class']].values.T
    y = y.astype('uint8')
    x = x.transpose()
    y = y.transpose()

    print("Database:", x)
    print("Target:", y)
    print("Target:", y.shape, x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize)

    return x_train, x_test, y_train, y_test
