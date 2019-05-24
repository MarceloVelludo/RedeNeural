import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Função usada para converter as opções de classe inteiras em um vetor binario.
def BinaryConverter(x):

    if x == 0:
        return np.array([1, 0, 0])
    if x == 1:
        return np.array([0, 1, 0])
    if x == 2:
        return np.array([0, 0, 1])

#Função que calcula a função limiar.
#No caso utilizamos uma relação direta entre
# a entrada e a saida como função limiar.
def functionLim(a):

    e_x = np.exp(a)
    return np.divide((e_x), np.sum(e_x))

#Função utilizada para retornar a classe que contem a maior chanceY.
def returnClass(chanceY):

    index = 0
    count = 0
    for i in chanceY:
        if i == None:
            index = count
        if i >= chanceY[index]:
            index = count
        count += 1

    return index


#Algoritmo de treinamento do Perceptron.
def perceptronTrain(max_it, alpha, x, d, x_teste, y_teste):
    print("\n**********************************\n")
    print("Inicio do treinamento.")
    print("Taxa de aprendizado:", alpha)
    ite = 1
    errorAcu = np.ones(shape=(1, 3))
    w = np.zeros(shape=(3, len(x[1])))
    b = np.ones(shape=(3, 1))
    print("W Inicial:\n", w)
    print("b Inicial:\n", b)
    errorAcuHist = np.zeros(0)

    while (ite < max_it) and (errorAcu != 0).any():
        errorAcu = np.zeros(shape=(1, 3))
        yR = np.zeros(0)
        for count in range(len(x[:])):
            xTrans = x[count].reshape(4, 1)
            aux = np.empty(0)
            i = 0

            for wi in w[:]:
                aux = np.append(aux, np.add(np.dot(wi, xTrans), b[i]))
                i += 1

            chanceY = functionLim(aux)
            y = returnClass(chanceY)
            yR = np.append(yR, y)
            erro = np.add(BinaryConverter(d[count]), (-BinaryConverter(y)))
            if (erro != 0).any():
                aux2 = np.multiply(erro.reshape(3, 1), x[count][:])
                aux2 = np.multiply(aux2, alpha)
                w = np.add(w, aux2)
                aux2 = np.multiply(alpha, erro.reshape(3, 1))
                b = np.add(b, aux2)
                errorAcu += erro**2

        perceptronTest(x_teste, y_teste, w, b)
        ite += 1
        errorAcuHist = np.append(errorAcuHist, (errorAcu.sum()/errorAcu.size))
        print("Época:\n", ite)
        print("Erro Acumulado:\n", errorAcu)
        print("Erro Quadrático Médio:", (errorAcu.sum()/errorAcu.size))

    fig, ax = plt.subplots()


    line1 = ax.plot(errorAcuHist, label='Erro Quadrático Médio')
    ax.legend()
    plt.show()
    print("Matrix Confusão:\n", confusion_matrix(d, yR))
    return w, b

def perceptronTest(x, d, w, b):
    correctAnswers = 0
    y = np.empty(0)

    for count in range(len(x[:])):
        xTransposto = x[count].reshape(4, 1)
        aux = np.dot(w, xTransposto)
        chanceY = functionLim(aux + b)
        y = np.append(y, returnClass(chanceY))

    for i in range(len(y)):
        if y[i] == d[i]:
           correctAnswers += 1
    print("Eficiencia da Rede Neural:\n", (correctAnswers/len(y))*100, "%")
    print("Matrix Confusão:\n", confusion_matrix(d, y))
    return