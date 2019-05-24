# coding: utf-8
import SplitData as sd
import Procedure as proc
import numpy as np


#Define o maximo de iterações
max_it = 1000
#define o alpha
alpha = 0.001

#Captura os dados do TXT que se encontra na pasta principal,
# divide a base em teste e treino, onde o primeiro paramento
# é o caminho para a base de dados, e o segundo parametro é a porcentagem que será concedida para testes e o resto irá para treinamento.
x_train, x_test, y_train, y_test = sd.splitData("iris.data", 0.30)
x_test, x_val, y_test, y_val = sd.splitTestInValidation(x_test, y_test, 0.5)


#Algoritmo de treinamento
w, b = proc.perceptronTrain(max_it, alpha, x_train, y_train, x_test, y_test)

print("\n**********************************\n")
print("Fim do Treinamento!")
print("W Final:\n", w)
print("b Final:\n", b)
print("Resultado aplicado a base de validação:")
#Resultados com a base de validação.
proc.perceptronTest(x_val, y_val, w, b)






