from gerenciador import Gerenciador
from vetorizador import CountVectorizer, TFIDFVectorizer
from classificador import NaiveBayes, SVM


classificador1 = NaiveBayes("naive_bayes_1_1.pickle", vetorizador1, recomendacoes)
classificador2 = NaiveBayes("naive_bayes_1_4.pickle", vetorizador2, recomendacoes)
classificador3 = SVM("svm_1_1.pickle", vetorizador3, recomendacoes)
classificador4 = SVM("svm_1_4.pickle", vetorizador4, recomendacoes)

print(classificador1.retorno())
print(classificador2.retorno())
print(classificador3.retorno())
print(classificador4.retorno())