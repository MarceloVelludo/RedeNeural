import numpy as np

#Função que calcula a função limiar.
#No caso utilizamos uma relação direta entre
# a entrada e a saida como função limiar.
def funcaoLimiar(a):
    return a

#Algoritmo Perceptron
def perceptronTrain(max_it, alpha, x, d, x_teste, y_teste):
    iteracao = 1
    erroAcumulado = 1
    y = np.zeros(shape=(len(d), 1))
    w = np.zeros(shape=(len(x[:][1]), 1))
    b = np.zeros(1)

    while (iteracao < max_it) and (erroAcumulado >0):

        print("Época:", iteracao)

        for count in range(len(x[:])):
            x_transposto = np.transpose(np.transpose(x[count]))
            x_teste = x[count]
            a = np.array([5, 4])[np.newaxis]
            print(a)
            print(a.T)
            aux = np.dot(w, x[count].transpose())
            y = funcaoLimiar(w*x[count].transpose() + b)
            erro = d[count][:] - y
            w += alpha*erro*x[count][:]
            b += alpha*erro
            erroAcumulado += erro^2

        perceptronTest(x_teste, y_teste, w, b)
        iteracao += 1


    return w, b

def perceptronTest(x, d, w, b):
    resultadoCorreto = 0
    count2 = 0

    for count in range(len(x[:])):
        y = funcaoLimiar(w * (x[count].transpose()) + b)

    for i in range(len(y)):
        if i == d[i]:
           resultadoCorreto += 1

    print("Eficiencia da Rede Neural: ", (resultadoCorreto/count2), "%")

    return