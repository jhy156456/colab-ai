import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_excel('news_data.xlsx', engine='openpyxl')


for i in df['jongcode'].unique():

    str_jong = "{}".format(i).split('.')[0]
    
    data = {
            'document' : df.loc[df['jongcode'] == i].document,
            'label' : df.loc[df['jongcode'] == i].label,
            'jongcode' : df.loc[df['jongcode'] == i].jongcode
            }
    train_data= pd.DataFrame(data,columns = ['document', 'label', 'jongcode'])
    test_data= pd.DataFrame(data,columns = ['document', 'label', 'jongcode'])
    
    train_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    train_data['document'] = train_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
    train_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    train_data = train_data.dropna(how='any') # Null 값 제거
    print('<종목코드', str_jong, '> 전처리 후 훈련용 샘플의 개수 :',len(train_data))
    
    test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
    test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any') # Null 값 제거
    print('<종목코드', str_jong, '> 전처리 후 테스트용 샘플의 개수 :',len(test_data))
    
    
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    
    
    print('<종목코드', str_jong, '> 훈련용 샘플 토큰화 중...')

    okt = Okt()
    X_train = []
    for sentence in train_data['document']:
        temp_X = okt.morphs(sentence, stem=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)


    print('<종목코드', str_jong, '> 테스트용 샘플 토큰화 중...')

    X_test = []
    for sentence in test_data['document']:
        temp_X = okt.morphs(sentence, stem=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)


    print('<종목코드', str_jong, '> 리뷰의 최대 길이 :',max(len(l) for l in X_train))
    print('<종목코드', str_jong, '> 리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))

    
    tokenizer = Tokenizer()
    print(X_train)
    tokenizer.fit_on_texts(X_train)
    #print(tokenizer.word_index)


    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('<종목코드', str_jong, '> 단어 집합(vocabulary)의 크기 :',total_cnt)
    print('<종목코드', str_jong, '> 등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
    print('<종목코드', str_jong, "> 단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
    print('<종목코드', str_jong, "> 전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
    
    
    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('<종목코드', str_jong, '> 단어 집합의 크기 :',vocab_size)


    tokenizer = Tokenizer(vocab_size) 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)


    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

    # 빈 샘플들을 제거
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    print('<종목코드', str_jong, '> 훈련용 빈샘플제거 후 개수',len(X_train))
    print('<종목코드', str_jong, '> 테스트용 빈샘플제거 후 개수', len(y_train))
    
    max_len = 30
    
    
   #  def below_threshold_len(max_len, nested_list):
   #      cnt = 0
   #      for s in nested_list:
   #          if(len(s) <= max_len):
   #              cnt = cnt + 1
   #      print('<종목코드', str_jong, '> 전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
   #
   # # below_threshold_len(max_len, X_train)
   #
   #
   #  X_train = pad_sequences(X_train, maxlen = max_len)
   #  X_test = pad_sequences(X_test, maxlen = max_len)
   #
   #
   #  from tensorflow.keras.layers import Embedding, Dense, LSTM
   #  from tensorflow.keras.models import Sequential
   #  from tensorflow.keras.models import load_model
   #  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
   #
   #  model = Sequential()
   #  model.add(Embedding(vocab_size, 100))
   #  model.add(LSTM(128))
   #  model.add(Dense(1, activation='sigmoid'))
   #
   #
   #  file_name = 'best_model_' + "{}".format(i).split('.')[0] + '.h5'
   #
   #  es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=10)
   #  #mc = ModelCheckpoint('best_model_samsung.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
   #  mc = ModelCheckpoint(file_name, monitor='val_acc', mode='max', verbose=1, save_best_only=False)
   #
   #
   #  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
   #
   #  print('<종목코드', str_jong, '>데이터 학습 중...')
   #
   #  history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
   #  #history = model.fit(X_train, y_train, epochs=13, batch_size=60, validation_split=0.2)
   #
   #  loaded_model = load_model(file_name)
   #  print('<종목코드', str_jong, "> 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
