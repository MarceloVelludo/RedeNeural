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
    e_x = np.exp(a - np.max(a))
    return e_x / e_x.sum()
    #return ((np.exp(a))/(np.exp(a*1)+np.exp(a*2)))

def returnClass(chanceY):

    indiceMaior = 0;
    count = 0
    print("chance:", chanceY)

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
            aux = np.dot(w, xTransposto)
            chanceY = funcaoLimiar(aux + b)
            y = returnClass(chanceY)
            #print("classe:", y)
            erro = int(d[count])-int(y)
            if erro != 0:
                aux2 = alpha*erro*x[count][:]
                #print("w antes:", w[y])
                w[y] = np.add(w[y], aux2)
                #print("w depois:", w[y])
                b[y] = np.add(b[y], alpha*erro)
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
    print("Resultado correto:", resultadoCorreto)
    print("len(y):", len(y))
    print("Eficiencia da Rede Neural: ", (resultadoCorreto/len(y))*100, "%")

    return