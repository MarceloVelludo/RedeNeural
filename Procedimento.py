import numpy as np

#Função que calcula a função limiar.
#No caso utilizamos uma relação direta entre
# a entrada e a saida como função limiar.

def conversorBinario(x):
    if x == 0:
        return np.array([1, 0, 0])
    if x == 1:
        return np.array([0, 1, 0])
    if x == 2:
        return np.array([0, 0, 1])


def funcaoLimiar(a):
    #if a < 1:
    #    return 0
    #if a >= 1 and a < 2:
    #    return 1
    #if a >= 2:
    #    return 2
    #return (1/(1+np.exp(-a)))
    #"""Compute softmax values for each sets of scores in x."""
    #e_x = np.exp(a - np.max(a))
    e_x = np.exp(a)
    #print("a:", a)
    #print("e_x: ", e_x)
    #print("np.sum(): ", np.sum(e_x))
    return np.divide((e_x), np.sum(e_x))
    #return ((np.exp(a))/(np.exp(a*1)+np.exp(a*2)))

def returnClass(chanceY):

    indiceMaior = 0;
    count = 0
    #print("chance:", chanceY)

    for i in chanceY:
        if i == None:
            indiceMaior = count
        if i >= chanceY[indiceMaior]:
            indiceMaior = count
        count += 1

    return indiceMaior


#Algoritmo Perceptron
def perceptronTrain(max_it, alpha, x, d, x_teste, y_teste):

    iteracao = 1
    erroAcumulado = np.ones(shape=(1, 3))
    y = 0
    w = np.zeros(shape=(3, len(x[1])))
    b = np.ones(shape=(3, 1))

    while (iteracao < max_it) and (erroAcumulado != 0).any():
        #print((erroAcumulado != 0).any())
        #print("erro:\n", np.max(erroAcumulado))
        erroAcumulado = np.zeros(shape=(1, 3))

        for count in range(len(x[:])):
            xTransposto = x[count].reshape(4, 1)
            #print("w:\n", w)
            #print("xTransposto:\n", xTransposto)
            aux = np.empty(0)
            i = 0
            #print("w x x':\n", np.dot(w, xTransposto))
            #print("b[i]:\n", b)
            for wi in w[:]:
                aux = np.append(aux, np.add(np.dot(wi, xTransposto), b[i]))
                i += 1
            #aux = np.dot(w, xTransposto)
            #aux = np.add(aux, b)

            chanceY = funcaoLimiar(aux)
            #print("chance:\n", chanceY)
            y = returnClass(chanceY)
            #print("y:\n", y)
            #print("classe:", y)
            erro = np.add(conversorBinario(d[count]), (-conversorBinario(y)))
            #print("erro:\n", erro)
            if (erro != 0).any():
                aux2 = np.multiply(erro.reshape(3, 1), x[count][:])
                aux2 = np.multiply(aux2, alpha)
                w = np.add(w, aux2)
                aux2 = np.multiply(alpha, erro.reshape(3, 1))
                b = np.add(b, aux2)
                #print("b[i]:\n", b)
                erroAcumulado += erro**2

        perceptronTest(x_teste, y_teste, w, b)
        iteracao += 1
        print("Época:\n", iteracao)
        #print("W:\n", w)
        #print("b:\n", b)
        print("erro Acumulado:\n", erroAcumulado)

    return w, b

def perceptronTest(x, d, w, b):
    resultadoCorreto = 0
    y = np.empty(0)

    for count in range(len(x[:])):
        xTransposto = x[count].reshape(4, 1)
        aux = np.dot(w, xTransposto)
        chanceY = funcaoLimiar(aux + b)
        y = np.append(y, returnClass(chanceY))

    for i in range(len(y)):
        if y[i] == d[i]:
           resultadoCorreto += 1
    #print("Resultado correto:", resultadoCorreto)
    #print("len(y):", len(y))
    print("Eficiencia da Rede Neural:\n", (resultadoCorreto/len(y))*100, "%")

    return