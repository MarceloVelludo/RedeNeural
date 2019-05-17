import numpy as np

#Função que calcula a função limiar.
#No caso utilizamos uma relação direta entre
# a entrada e a saida como função limiar.
def funcaoLimiar(a):
    return a

#Algoritmo Perceptron
def perceptronTrain(max_it, alpha, x, d):
    iteracao = 1
    erroAcumulado = 1
    y = np.zeros(shape=(len(d), 1))
    w = np.zeros(shape=(len(x[:], 1)))
    b = np.zeros(shape=(len(x[:], 1)))

    while (iteracao < max_it) and (erroAcumulado >0):

        for count in len(x[:]):
            y = funcaoLimiar(w*(x[count].transpose()) + b)
            erro = d[count][:] - y
            w += alpha*erro*x[count][:]
            b += alpha*erro
            erroAcumulado += erro^2

        iteracao += 1

    return w,b

def perceptronTest(x, d, w, b):
    resultadoCorreto = 0
    count2 = 0

    for count in len(x[:]):
        y = funcaoLimiar(w * (x[count].transpose()) + b)

    for i in y:
        count2 += 1
        if i == d[count2]:
           resultadoCorreto += 1

    print("Eficiencia da Rede Neural: ", (resultadoCorreto/count2), "%")

    return