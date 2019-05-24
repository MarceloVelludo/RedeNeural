import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#Função que separa e organiza os dados em dois datas set: Train e Test
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

    return np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)

#Função que separa o set test em test e validação
def splitTestInValidation(x,y,test_size):

    x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=test_size)

    return np.asarray(x_test), np.asarray(x_val), np.asarray(y_test), np.asarray(y_val)