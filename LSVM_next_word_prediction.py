import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
import platform

def limpar_terminal():
    sistema_operacional = platform.system()

    if sistema_operacional == 'Linux':
        os.system('clear')
    elif sistema_operacional == 'Windows':
        os.system('cls')
    else:
        print("Sistema operacional não suportado.")

modelo1 = load_model('modelo_LSVM.h5')
data1 =  open("datasetTextosEmPortugues.txt", encoding="utf-8")
meuTexto1 = data1.read()
mytokenizer1 = Tokenizer()
mytokenizer1.fit_on_texts([meuTexto1])
total_words1 = len(mytokenizer1.word_index) + 1
my_input_sequences1 = []
for line in meuTexto1.split('\n'):
    #print(line)
    token_list1 = mytokenizer1.texts_to_sequences([line])[0]
    #print(token_list)
    for i in range(1, len(token_list1)):
        my_n_gram_sequence1 = token_list1[:i+1]
        #print(my_n_gram_sequence)
        my_input_sequences1.append(my_n_gram_sequence1)
        #print(input_sequences)
max_sequence_len1 = max([len(seq) for seq in my_input_sequences1])
input_sequences1 = np.array(pad_sequences(my_input_sequences1, maxlen=max_sequence_len1, padding='pre'))
X1 = input_sequences1[:, :-1]
y1 = input_sequences1[:, -1]
y1 = np.array(tf.keras.utils.to_categorical(y1, num_classes=total_words1))

modelo2 = load_model('modelo_LSVM2.h5')
data2 =  open("sentencas_sem_duplicatas.txt", encoding="utf-8")
meuTexto2 = data2.read()
mytokenizer2 = Tokenizer()
mytokenizer2.fit_on_texts([meuTexto2])
total_words2 = len(mytokenizer2.word_index) + 1
my_input_sequences2 = []
for line in meuTexto2.split('\n'):
    #print(line)
    token_list2 = mytokenizer2.texts_to_sequences([line])[0]
    #print(token_list)
    for i in range(1, len(token_list2)):
        my_n_gram_sequence2 = token_list2[:i+1]
        #print(my_n_gram_sequence)
        my_input_sequences2.append(my_n_gram_sequence2)
        #print(input_sequences)
max_sequence_len2 = max([len(seq) for seq in my_input_sequences2])
input_sequences2 = np.array(pad_sequences(my_input_sequences2, maxlen=max_sequence_len2, padding='pre'))
X2 = input_sequences2[:, :-1]
y2 = input_sequences2[:, -1]
y2 = np.array(tf.keras.utils.to_categorical(y2, num_classes=total_words2))

def preverProxPalavra(frase, nPalavras, modelo):
    if(modelo == 1):
        for _ in range(nPalavras):
            token_list = mytokenizer1.texts_to_sequences([frase])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len1-1, padding='pre')
            predicted = np.argmax(modelo1.predict(token_list), axis=-1)
            output_word = ""
            for word, index in mytokenizer1.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            frase += " " + output_word
        return frase

    if(modelo == 2):
        for _ in range(nPalavras):
            token_list = mytokenizer2.texts_to_sequences([frase])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len2-1, padding='pre')
            predicted = np.argmax(modelo2.predict(token_list), axis=-1)
            output_word = ""
            for word, index in mytokenizer1.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            frase += " " + output_word
        return frase
    


def main():
    controle = True
    menu = 0

    modelo = 1

    while controle:
        print("Usando modelo " + str(modelo) + "\n" + 
              "1- Prever próxima palavra iterativo\n" +
              "2- Prever N próximas palavras\n" +
              "3- Alterar modelo")
        
        menu = input("Digite sua opção: ")

        if int(menu) == 1:
            limpar_terminal()
            prevendo = True
            print("\nDigite enter e aperte enter para perver. (exit para sair)\n")
            frase = ""
            while prevendo:
                print(frase, end=" ")
                frase = frase + " " + input()
                if("exit" in frase):
                    prevendo = False
                frase = preverProxPalavra(frase, 1, 1)
                limpar_terminal()

        if int(menu) == 2:
            aux = True
            while aux:
                limpar_terminal()
                frase = input("Digite a frase incial (exit para sair): ")
                nPalavras = input("Digite o número de palavras que deseja prever: ")
                print(preverProxPalavra(frase, int(nPalavras), modelo))
                menu = 0 
                if(exit in frase):
                    aux = False
                    
        if int(menu) == 3:
            limpar_terminal()
            if(modelo == 1):
                modelo = 2
            elif(modelo == 2):
                modelo = 1
            print("Modelo alterado")
            
        elif menu not in [0, 1, 2, 3]:
            print("Opção inválida")
            menu = 0

main()