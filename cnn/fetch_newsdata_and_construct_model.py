import math
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from keras.layers import Dropout, Flatten
from keras.models import load_model
from konlpy.tag import Okt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def fetch_newsdate_and_construct_model(seq_len, epochs, batch_size, symbol, output_filename, isMultiInput):
    start_time = time.monotonic()
    training_filename = "./stockdatas/" + symbol + "_news_training.csv"
    testing_filename = "./stockdatas/" + symbol + "_news_testing.csv"
    news_train_data = pd.read_csv(training_filename)
    news_test_data = pd.read_csv(testing_filename)

    news_sequence_train_data = pd.DataFrame(columns=['Content', 'label'])
    news_sequence_test_data = pd.DataFrame(columns=['Content', 'label'])
    print(seq_len)
    if not isMultiInput:
        for i in range(0, len(news_train_data)):
            c = news_train_data.iloc[i:i + int(seq_len) + 1, :]
            if len(c) == int(seq_len) + 1:
                series_to_string = ' '.join(c["Content"].apply(str))
                series_to_string = series_to_string.replace('nan','')
                label = c["label"].iloc[-1]
                news_sequence_train_data = news_sequence_train_data.append(
                    pd.DataFrame([{'Content':series_to_string, 'label':label}]))

        for i in range(0, len(news_test_data)):
            c = news_test_data.iloc[i:i + int(seq_len) + 1, :]
            if len(c) == int(seq_len) + 1:
                series_to_string = ' '.join(c["Content"].apply(str))
                series_to_string = series_to_string.replace('nan', '')
                label = c["label"].iloc[-1]
                news_sequence_test_data = news_sequence_test_data.append(
                    pd.DataFrame([(series_to_string, label)], columns=['Content', 'label']), ignore_index=True)

    print("------------------------------")
    print("news train size : ", len(news_sequence_train_data))
    print("------------------------------")
    print("news train size : ", len(news_sequence_test_data))
    print("------------------------------")



    df_train_data = {
        'document': news_train_data.Content,
        'label': news_train_data.label
    }

    df_test_data = {
        'document': news_test_data.Content,
        'label': news_test_data.label
    }

    train_data = pd.DataFrame(df_train_data, columns=['document', 'label'])
    test_data = pd.DataFrame(df_test_data, columns=['document', 'label'])

    train_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
    train_data['document'] = train_data['document'].str.replace('^ +', "")  # 공백은 empty 값으로 변경
    train_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
    train_data = train_data.dropna(how='any')  # Null 값 제거
    print('<종목코드', symbol, '> 전처리 후 훈련용 샘플의 개수 :', len(train_data))

    test_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
    test_data['document'] = test_data['document'].str.replace('^ +', "")  # 공백은 empty 값으로 변경
    test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any')  # Null 값 제거
    print('<종목코드', symbol, '> 전처리 후 테스트용 샘플의 개수 :', len(test_data))

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    print('<종목코드', symbol, '> 훈련용 샘플 토큰화 중...')

    okt = Okt()
    news_X_train = []
    for sentence in train_data['document']:
        temp_X = okt.morphs(sentence, stem=True)  # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
        # temp_X = np.asarray(temp_X)
        news_X_train.append(temp_X)
    print('<종목코드', symbol, '> 테스트용 샘플 토큰화 중...')

    news_X_test = []
    for sentence in test_data['document']:
        temp_X = okt.morphs(sentence, stem=True)  # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
        # temp_X = np.asarray(temp_X)
        news_X_test.append(temp_X)

    print('<종목코드', symbol, '> 리뷰의 최대 길이 :', max(len(l) for l in news_X_train))
    print('<종목코드', symbol, '> 리뷰의 평균 길이 :', sum(map(len, news_X_train)) / len(news_X_train))

    tokenizer = Tokenizer()
    print("------------------------------------------------------------------------------------------")
    print("news_X_train.size() : ", len(news_X_train))
    print("------------------------------------------------------------------------------------------")
    print("news_X_test.size() : ", len(news_X_test))
    print("------------------------------------------------------------------------------------------")

    # fit_on_texts : 문자 데이터를 입력받아서 리스트의 형태로 변환합니다.
    tokenizer.fit_on_texts(news_X_train)
    # '단어' : 정수 형태로 반환한다.
    print(tokenizer.word_index)

    threshold = 3
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('<종목코드', symbol, '> 단어 집합(vocabulary)의 크기 :', total_cnt)
    print('<종목코드', symbol, '> 등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
    print('<종목코드', symbol, "> 단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print('<종목코드', symbol, "> 전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('<종목코드', symbol, '> 단어 집합의 크기 :', vocab_size)

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(news_X_train)
    # print("------------------------------------------------------------------------------------------")
    # print(news_X_train)
    # print("------------------------------------------------------------------------------------------")
    # 텍스트 안의 단어들을 숫자의 시퀀스의 형태로 변환합니다.
    # 대한항공:1, 체결:2, 실적:3 ....
    # [주리다,탈출,비법,공개,기관,억...]
    # -> [9,63,26,5]
    # 이 행위는 단어ID 표현인듯!
    # 여기서 to_categorical()을 사용하면 원핫인코딩이 된다
    # https://dacon.io/en/codeshare/1839
    news_X_train = tokenizer.texts_to_sequences(news_X_train)
    news_Y_train = np.array(train_data['label'])

    print("뉴스 훈련 사이즈 : ",len(news_X_train))

    news_X_test = tokenizer.texts_to_sequences(news_X_test)

    print("뉴스 테스트 사이즈 : ", len(news_X_test))
    news_Y_test = np.array(test_data['label'])

    # print("------------------------------------------------------------------------------------------")
    # print("texts_to_sequences")
    # print(news_X_train)
    # print("------------------------------------------------------------------------------------------")

    drop_train = [index for index, sentence in enumerate(news_X_train) if len(sentence) < 1]

    # 빈 샘플들을 제거
    news_X_train = np.delete(news_X_train, drop_train, axis=0)
    news_Y_train = np.delete(news_Y_train, drop_train, axis=0)
    print("------------------------------------------------------------------------------------------")

    # for i in range(len(news_X_train)):
    #     news_X_train[i] = np.array(news_X_train[i])
    # for i in range(len(news_Y_train)):
    #     news_Y_train[i] = np.array(news_Y_train[i])
    print(news_X_train[:3])
    print("------------------------------------------------------------------------------------------")
    print('<종목코드', symbol, '> 훈련용 빈샘플제거 후 개수', len(news_X_train))
    print('<종목코드', symbol, '> 테스트용 빈샘플제거 후 개수', len(news_Y_train))

    max_len = 100

    def below_threshold_len(max_len, nested_list):
        cnt = 0
        for s in nested_list:
            if (len(s) <= max_len):
                cnt = cnt + 1
        print('<종목코드', symbol, '> 전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (cnt / len(nested_list)) * 100))

    # below_threshold_len(max_len, news_X_train)

    # study
    # 서로 다른 개수의 단어로 이루어진 문장을 같은 길이로 만들어주기 위해 패딩을 사용할 수 있습니다.
    # 데이터가 maxlen 보다 길면 데이터를 자름
    # pre : 데이터 앞에 0으로 채움
    # post : 데이터 뒤에 0으로 채움
    # 패딩을 사용하기 위해서는 tensorflow.keras.preprocessing.sequence 모듈의 pad_sequences 함수를 사용합니다.
    # https://codetorial.net/tensorflow/natural_language_processing_in_tensorflow_01.html
    news_X_train = pad_sequences(news_X_train, maxlen=max_len)
    # print("------------------------------------------------------------------------------------------")
    # print("pad_sequences")
    # print(news_X_train)
    # print("------------------------------------------------------------------------------------------")
    news_X_test = pad_sequences(news_X_test, maxlen=max_len)

    from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import confusion_matrix

    # news_model = Sequential()
    # news_model.add(Embedding(vocab_size, 100))
    # news_model.add(Bidirectional(LSTM(128)))
    # # news_model.add(LSTM(128))
    # news_model.add(Dense(1, activation='sigmoid'))

    news_model = Sequential()
    # embedding : (input_dim = num_words(데이터 갯수))
    news_model.add(Embedding(vocab_size, 128))
    news_model.add(Bidirectional(LSTM(256)))
    news_model.add(Dense(128, activation="relu"))
    news_model.add(Dense(2, activation="softmax"))

    #
    # news_model.add(Bidirectional(LSTM(128)))
    # news_model.add(Bidirectional(LSTM(128)))
    # news_model.add(Bidirectional(LSTM(128)))
    # news_model.add(Dropout(0.3))
    # news_model.add(Bidirectional(LSTM(64)))
    # news_model.add(Bidirectional(LSTM(64)))
    # news_model.add(Bidirectional(LSTM(64)))
    # news_model.add(Dropout(0.3))
    # news_model.add(Bidirectional(LSTM(32)))
    # news_model.add(Bidirectional(LSTM(32)))
    # news_model.add(Bidirectional(LSTM(32)))
    # news_model.add(Dropout(0.3))
    # news_model.add(Bidirectional(LSTM(16)))
    # news_model.add(Bidirectional(LSTM(16)))
    # news_model.add(Bidirectional(LSTM(16)))
    # news_model.add(Dropout(0.3))
    # news_model.add(Bidirectional(LSTM(8)))
    # news_model.add(Bidirectional(LSTM(8)))
    # news_model.add(Bidirectional(LSTM(8)))
    # news_model.add(Dropout(0.3))
    # news_model.add(Bidirectional(LSTM(4)))
    # news_model.add(Bidirectional(LSTM(4)))
    # news_model.add(Bidirectional(LSTM(4)))
    # news_model.add(Flatten())
    # news_model.add(Dense(2, activation='softmax'))  # 3가지 클래스로 분류하고, 다중분류이므로 softmax함수를 활용한다.

    file_name = 'news{}epochs_{}batch_lstm_model_{}.h5'.format(epochs, batch_size, symbol)

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
    # mc = ModelCheckpoint('best_model_samsung.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    mc = ModelCheckpoint(file_name, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    news_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    print('<종목코드', symbol, '>데이터 학습 중...')

    from keras.utils import np_utils

    news_X_train = np.asarray(news_X_train).astype("float32")
    news_Y_train = np.asarray(news_Y_train).astype("float32")
    news_Y_train = np_utils.to_categorical(news_Y_train, 2)
    news_X_test = np.asarray(news_X_test).astype("float32")
    news_Y_test = np.asarray(news_Y_test).astype("float32")
    news_Y_test = np_utils.to_categorical(news_Y_test, 2)

    history = news_model.fit(news_X_train, news_Y_train, epochs=epochs, callbacks=[es, mc], batch_size=batch_size,
                             validation_split=0.2)

    """
    imdb.data_load()의 파라미터로 num_words를 사용하면 이 데이터에서 등장 빈도 순위로 몇 번째에 해당하는 단어까지를 사용할 것인지를 의미합니다.
    https://wikidocs.net/24586
    hist_dict = history.history
    hist_dict.keys()
    >> dict_keys(['loss','acc',..])
    https://www.youtube.com/watch?v=hR8Rvp-YNGg&ab_channel=%EC%9D%B4%EC%88%98%EC%95%88%EC%BB%B4%ED%93%A8%ED%84%B0%EC%97%B0%EA%B5%AC%EC%86%8C
    """
    # history = news_model.fit(news_X_train, news_Y_train, epochs=13, batch_size=60, validation_split=0.2)

    loaded_model = load_model(file_name)
    news_predicted = loaded_model.predict(news_X_test)

    news_predicted = np.argmax(news_predicted, axis=1)
    news_Y_test = np.argmax(news_Y_test, axis=1)
    # news_Y_test = np.argmax(news_Y_test, axis=1)
    print("------------------------------------------------------------------------------------------")
    print(news_predicted)
    print(news_Y_test)
    print(sklearn.metrics.accuracy_score(news_Y_test, news_predicted))
    print("------------------------------------------------------------------------------------------")

    if not isMultiInput:
        cm = confusion_matrix(news_Y_test, news_predicted)
        report = classification_report(news_Y_test, news_predicted)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        if tp == 0:
            tp = 1
        if tn == 0:
            tn = 1
        if fp == 0:
            fp = 1
        if fn == 0:
            fn = 1
        TPR = float(tp) / (float(tp) + float(fn))
        FPR = float(fp) / (float(fp) + float(tn))
        accuracy = round((float(tp) + float(tn)) / (float(tp) +
                                                    float(fp) + float(fn) + float(tn)), 3)
        specitivity = round(float(tn) / (float(tn) + float(fp)), 3)
        sensitivity = round(float(tp) / (float(tp) + float(fn)), 3)
        mcc = round((float(tp) * float(tn) - float(fp) * float(fn)) / math.sqrt(
            (float(tp) + float(fp))
            * (float(tp) + float(fn))
            * (float(tn) + float(fp))
            * (float(tn) + float(fn))
        ), 3)

        f_output = open(output_filename.output, 'a')
        f_output.write('=======\n')
        f_output.write('chart {}epochs_{}batch_resnet\n'.format(
            epochs, batch_size))
        f_output.write('TN: {}\n'.format(tn))
        f_output.write('FN: {}\n'.format(fn))
        f_output.write('TP: {}\n'.format(tp))
        f_output.write('FP: {}\n'.format(fp))
        f_output.write('TPR: {}\n'.format(TPR))
        f_output.write('FPR: {}\n'.format(FPR))
        f_output.write('accuracy: {}\n'.format(accuracy))
        f_output.write('specitivity: {}\n'.format(specitivity))
        f_output.write("sensitivity : {}\n".format(sensitivity))
        f_output.write("mcc : {}\n".format(mcc))
        f_output.write("{}".format(report))
        f_output.write('=======\n')
        f_output.close()
        end_time = time.monotonic()
        print("Duration : {}".format(timedelta(seconds=end_time - start_time)))
