import numpy as np

#Função que calcula a função limiar.
#No caso utilizamos uma relação direta entre
# a entrada e a saida como função limiar.
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
        if i >= chanceY[indiceMaior]:
            indiceMaior = count
        count += 1

    return indiceMaior


#Algoritmo Perceptron
def perceptronTrain(max_it, alpha, x, d, x_teste, y_teste):

    iteracao = 1
    erroAcumulado = 1.0
    y = np.zeros(shape=(1, 1))
    w = np.zeros(shape=(3, len(x[1])))
    b = np.zeros(shape=(3, 1))

    while (iteracao < max_it) and (erroAcumulado > 0.01):
        erroAcumulado = 0.0

        for count in range(len(x[:])):
            xTransposto = x[count].reshape(4, 1)
            aux = np.empty(0)
            i=0
            for wi in w[:]:
                aux = np.append(aux, np.add(np.dot(wi, xTransposto), b[i]))
                i+= 1
            #aux = np.dot(w, xTransposto)
            #aux = np.add(aux, b)
            chanceY = funcaoLimiar(aux)
            print("chance:", chanceY)
            y = returnClass(chanceY)
            print("y:", y)
            #print("classe:", y)
            erro = np.add(d[count], (-y))
            if erro != 0:
                aux2 = np.multiply(erro, x[count][:])
                aux2 = np.multiply(aux2, alpha)
                w[y] = np.add(w[y], aux2)
                aux2 = np.multiply(alpha, erro)
                b[y] = np.add(b[y], aux2)
                erroAcumulado += erro**2

        perceptronTest(x_teste, y_teste, w, b)
        iteracao += 1
        print("Época:", iteracao)
        print("W:", w)
        print("b:", b)
        print("erro Acumulado:", erroAcumulado)

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
    print("Eficiencia da Rede Neural: ", (resultadoCorreto/len(y))*100, "%")

    return