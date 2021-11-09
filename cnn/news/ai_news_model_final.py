import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from konlpy.tag import Okt
from pandas import DataFrame
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

str_jong = "272210.KS.xlsx"
symbol = "272210.KS"
df = pd.read_excel(str_jong, engine='openpyxl')
# 엑셀은 최신부터 내림차순인데 야후 데이터셋은 과거부터 오름차순이므로 역순으로 뒤집는다
df = df.iloc[::-1]
df["document"] = ""
df["label"] = 0
for i in range(len(df)):
    df["document"][i] = str(df.TITLE[i]) + str(df.CODE[i])

training_start_date = "2000-01-01"
training_end_date = "2021-05-31"
testing_start_date = "2021-06-01"
testing_end_date = "2021-12-31"
seq_len = 50
epochs = 150

# str_jong = "{}".format(i).split('.')[0]

newsTrainData = DataFrame(columns=("document", "label"))
newsTestData = DataFrame(columns=("document", "label"))
fname = "../stockdatas/" + symbol + "_training.csv"
stockDataDf = pd.read_csv(fname)
newsDataFrame = DataFrame(columns=("Date", "document", "label"))

# 야후 파이낸스에서 다운받은 주식 데이터와 일치하는 일들로만 뉴스 데이터를 셋팅한다.
for i in range(len(stockDataDf)):

    # print(stockDataDf.iloc[i])
    # print(data)
    # print(newsDataFrame.head())
    documentColumn = df[df['NEWS_DATE'] == stockDataDf.iloc[i]['Date']].document
    str = ""
    for index, item in enumerate(documentColumn):
        str += " " + item
    data = {
        'Date': stockDataDf.iloc[i]['Date'],
        'document': str,
        'label': 0
    }
    newsDataFrame = newsDataFrame.append(data, ignore_index=True)

pddf = pd.read_csv('../stockdatas/' + symbol + '_training.csv', parse_dates=True, index_col=0)
pddf.fillna(0)
pddf.reset_index(inplace=True)
# pddf['Date'] = pddf['Date'].map(mdates.date2num)
# df['NEWS_DATE'] = df['NEWS_DATE'].map(mdates.date2num)
for i in range(0, len(pddf)):
    # 수정!!  int(seq_len)+1 ->  int(seq_len)

    # 1일~seq_len+1 일치 데이터프레임 획득
    c = pddf.iloc[i:i + int(seq_len) + 1, :]
    starting = 0
    endvalue = 0
    label = ""

    if len(c) == int(seq_len) + 1:
        sliceNewsDataFrame = newsDataFrame.iloc[i:i + int(seq_len), :]
        starting = c["Close"].iloc[-2]
        endvalue = c["Close"].iloc[-1]
        tmp_rtn = endvalue / starting - 1
        if tmp_rtn > 0:
            # 상승
            label = 1
        else:
            # 하락
            label = 0

        str = ""
        for index, item in enumerate(sliceNewsDataFrame.document):
            str += " " + item
        data = {
            'document': str,
            'label': label
        }
        newsTrainData = newsTrainData.append(data, ignore_index=True)
        # study
        # https://stackoverflow.com/questions/17995328/changing-values-in-pandas-dataframe-does-not-work

print("training labeling end")
pddf = pd.read_csv('../stockdatas/' + symbol + '_testing.csv', parse_dates=True, index_col=0)
pddf.fillna(0)
pddf.reset_index(inplace=True)
# pddf['Date'] = pddf['Date'].map(mdates.date2num)
# df['NEWS_DATE'] = df['NEWS_DATE'].map(mdates.date2num)
for i in range(0, len(pddf)):
    # 수정!!  int(seq_len)+1 ->  int(seq_len)
    # 1일~seq_len+1 일치 데이터프레임 획득
    c = pddf.iloc[i:i + int(seq_len) + 1, :]
    starting = 0
    endvalue = 0
    label = ""
    if len(c) == int(seq_len) + 1:
        sliceNewsDataFrame = newsDataFrame.iloc[i:i + int(seq_len), :]
        starting = c["Close"].iloc[-2]
        endvalue = c["Close"].iloc[-1]
        tmp_rtn = endvalue / starting - 1
        if tmp_rtn > 0:
            # 상승
            label = 1
        else:
            # 하락
            label = 0
        # study
        # https://stackoverflow.com/questions/17995328/changing-values-in-pandas-dataframe-does-not-work
        # df.loc[df['NEWS_DATE'] == c.iloc[-1]["Date"], 'label'] = label
        str = ""
        for index, item in enumerate(sliceNewsDataFrame.document):
            str += " " + item
        data = {
            'document': str,
            'label': label
        }
        newsTestData = newsTestData.append(data, ignore_index=True)
print("testing labeling end")
print("countTrain : ", len(newsTrainData))
print("countTest : ", len(newsTestData))

df_train_data = {
    'document': newsTrainData.document,
    'label': newsTrainData.label
}

df_test_data = {
    'document': newsTestData.document,
    'label': newsTestData.label
}

train_data = pd.DataFrame(df_train_data, columns=['document', 'label'])
test_data = pd.DataFrame(df_test_data, columns=['document', 'label'])

train_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
train_data['document'] = train_data['document'].str.replace('^ +', "")  # 공백은 empty 값으로 변경
train_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
train_data = train_data.dropna(how='any')  # Null 값 제거
print('<종목코드', str_jong, '> 전처리 후 훈련용 샘플의 개수 :', len(train_data))

test_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "")  # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거
print('<종목코드', str_jong, '> 전처리 후 테스트용 샘플의 개수 :', len(test_data))

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

print('<종목코드', str_jong, '> 훈련용 샘플 토큰화 중...')

okt = Okt()
news_X_train = []
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    # temp_X = np.asarray(temp_X)
    news_X_train.append(temp_X)
print('<종목코드', str_jong, '> 테스트용 샘플 토큰화 중...')

news_X_test = []
for sentence in test_data['document']:
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    # temp_X = np.asarray(temp_X)
    news_X_test.append(temp_X)

print('<종목코드', str_jong, '> 리뷰의 최대 길이 :', max(len(l) for l in news_X_train))
print('<종목코드', str_jong, '> 리뷰의 평균 길이 :', sum(map(len, news_X_train)) / len(news_X_train))

tokenizer = Tokenizer()
print("------------------------------------------------------------------------------------------")
print("news_X_train.size() : ", len(news_X_train))
print("------------------------------------------------------------------------------------------")
print("news_X_test.size() : ", len(news_X_test))
print("------------------------------------------------------------------------------------------")

# fit_on_texts : 문자 데이터를 입력받아서 리스트의 형태로 변환합니다.
tokenizer.fit_on_texts(news_X_train)
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

print('<종목코드', str_jong, '> 단어 집합(vocabulary)의 크기 :', total_cnt)
print('<종목코드', str_jong, '> 등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print('<종목코드', str_jong, "> 단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print('<종목코드', str_jong, "> 전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('<종목코드', str_jong, '> 단어 집합의 크기 :', vocab_size)

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
news_X_test = tokenizer.texts_to_sequences(news_X_test)
# print("------------------------------------------------------------------------------------------")
# print("texts_to_sequences")
# print(news_X_train)
# print("------------------------------------------------------------------------------------------")

news_Y_train = np.array(train_data['label'])
news_Y_test = np.array(test_data['label'])

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
print('<종목코드', str_jong, '> 훈련용 빈샘플제거 후 개수', len(news_X_train))
print('<종목코드', str_jong, '> 테스트용 빈샘플제거 후 개수', len(news_Y_train))

max_len = 30


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt + 1
    print('<종목코드', str_jong, '> 전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (cnt / len(nested_list)) * 100))


# below_threshold_len(max_len, news_X_train)

# study
# 서로 다른 개수의 단어로 이루어진 문장을 같은 길이로 만들어주기 위해 패딩을 사용할 수 있습니다.
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

news_model = Sequential()
news_model.add(Embedding(vocab_size, 100))
news_model.add(Bidirectional(LSTM(128)))
# news_model.add(LSTM(128))
news_model.add(Dense(1, activation='sigmoid'))

file_name = 'best_model_' + "{}".format(i).split('.')[0] + '.h5'

es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=10)
# mc = ModelCheckpoint('best_model_samsung.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
mc = ModelCheckpoint(file_name, monitor='val_acc', mode='max', verbose=1, save_best_only=False)

news_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print('<종목코드', str_jong, '>데이터 학습 중...')


news_X_train = np.asarray(news_X_train).astype("float32")
news_Y_train = np.asarray(news_Y_train).astype("float32")
news_X_test = np.asarray(news_X_test).astype("float32")
news_Y_test = np.asarray(news_Y_test).astype("float32")


history = news_model.fit(news_X_train, news_Y_train, epochs=epochs, callbacks=[es, mc], batch_size=60,
                         validation_split=0.2)
# history = news_model.fit(news_X_train, news_Y_train, epochs=13, batch_size=60, validation_split=0.2)

news_predicted = news_model.predict(news_X_test)

news_Y_pred = np.argmax(news_predicted, axis=1)
print("------------------------------------------------------------------------------------------")
print(sklearn.metrics.accuracy_score(news_Y_test, news_Y_pred))
print("------------------------------------------------------------------------------------------")


def plot_acc_loss_epoch(history):
    # https://snowdeer.github.io/machine-learning/2018/01/10/check-relation-between-epoch-and-loss-using-graph/
    y_acc = history.history['acc']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_acc, marker='.', c='red', label="accuracy")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# print(history.history.keys())
# plot_acc_loss_epoch(history)
