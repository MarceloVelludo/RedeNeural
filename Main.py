import SplitData as sd
import Procedimento as proc



try:
    #Captura os dados do TXT que se encontra na pasta principal,
    # divide a base em teste e treino, onde o primeiro paramento
    # é o caminho para a base de dados, e o segundo parametro é a porcentagem que será concedida para testes e o resto irá para treinamento.
    x_train, x_test, y_train, y_test = sd.splitData("iris.data", 0.25)

    #Algoritmo de treinamento
    w = perceptron(max_it, alpha, x_train, y_test)

except:
    print("Something went wrong")




