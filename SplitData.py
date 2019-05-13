import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def splitData(caminho, testSize):

    test = pd.read_csv(caminho, header=None)
    test.columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

    database = test.loc[:, "sepal-length":"petal-width"]
    target = test['class'][:]

    print("Database:", database)
    print("Target:", target)

    x_train, x_test, y_train, y_test = train_test_split(database, target, test_size= testSize)

    return x_train, x_test, y_train, y_test
