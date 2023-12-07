import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from flask import Flask, jsonify, request
from keras.models import load_model

def preverProxPalavra(frase, nPalavras, modelo):
    if(modelo == 1):
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

        for _ in range(nPalavras):
            token_list = mytokenizer1.texts_to_sequences([frase])[0]
            print(token_list)
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

        for _ in range(nPalavras):
            token_list = mytokenizer2.texts_to_sequences([frase])[0]
            print(token_list)
            token_list = pad_sequences([token_list], maxlen=max_sequence_len2-1, padding='pre')
            predicted = np.argmax(modelo2.predict(token_list), axis=-1)
            output_word = ""
            for word, index in mytokenizer2.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            frase += " " + output_word
        return frase

app = Flask(__name__)

# Rota para receber o JSON e retornar uma resposta
@app.route('/prever-palavra', methods=['POST'])
def processar_string():
    # Verifica se o conteúdo recebido é um JSON
    if request.is_json:
        # Recebe o JSON enviado na requisição
        json_data = request.get_json()
        
        # Verifica se a chave 'string' está presente no JSON
        if 'frase' in json_data:
            # Obtém a string do JSON
            frase = json_data['frase']
            nPalavra = json_data['nPalavra']
            modelo = json_data['modelo']
            
            # Aqui você pode realizar qualquer processamento necessário na string
            # Neste exemplo, apenas adicionamos um prefixo à string recebida
            frasePrevista = preverProxPalavra(frase, nPalavra, modelo);
            
            # Cria um novo JSON com a string processada e o retorna
            response = {
                "frasePrevista": frasePrevista
            }
            
            # Retorna a resposta em formato JSON
            return jsonify(response), 200
        
        else:
            # Se a chave 'string' não estiver presente no JSON
            return jsonify({"error": "Chave 'string' não encontrada"}), 400
    
    else:
        # Se o conteúdo recebido não for um JSON
        return jsonify({"error": "Conteúdo não é um JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True)
