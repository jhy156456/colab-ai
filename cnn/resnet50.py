import os
import logging
from collections import Counter

from eunjeon import Mecab
from sklearn.model_selection import train_test_split

logging.disable(logging.WARNING)
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings

warnings.filterwarnings(action='ignore')
# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import math

import keras
from keras.layers import Input, Dense, Conv2D, ZeroPadding2D, Flatten, Activation, add, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import np_utils
from tensorflow.keras.optimizers import *
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import argparse
import time
from datetime import timedelta
from utils.dataset import dataset


def build_dataset(data_directory, img_width):
    X, y, tags = dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_model(SHAPE, nb_classes, bn_axis, seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    x = ZeroPadding2D((3, 3))(input_layer)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # [배열 안은 필터의갯수]
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # print("x nya {}".format(x))
    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(10, activation='relu', name='fc10')(x)

    model = Model(input_layer, x)

    return model


def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=48)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    # parser.add_argument('-o', '--optimizer',
    #                     help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="hasilnya.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if keras.backend.image_data_format() == 'tf' else 1

    data_directory = args.input
    period_name = data_directory.split('/')

    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), args.dimension)
    print("------------------------------------------------------------------------------------------")
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), args.dimension)
    print("------------------------------------------------------------------------------------------")
    print("number of classes : {}".format(nb_classes))

    model = build_model(SHAPE, nb_classes, bn_axis)
    #################################뉴스 데이터 추출 및 병합끝############################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn.metrics
    from konlpy.tag import Okt
    from pandas import DataFrame
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    str_jong = "./news/272210.KS.xlsx"
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
    seq_len = 30
    # epochs = 150

    # str_jong = "{}".format(i).split('.')[0]

    newsTrainData = DataFrame(columns=("document", "label"))
    newsTestData = DataFrame(columns=("document", "label"))
    fname = "./stockdatas/" + symbol + "_training.csv"
    stockDataDf = pd.read_csv(fname)
    newsDataFrame = DataFrame(columns=("Date", "document", "label"))

    # 야후 파이낸스에서 다운받은 주식 데이터와 일치하는 일들로만 뉴스 데이터를 셋팅한다.
    for i in range(len(stockDataDf)):

        # print(stockDataDf.iloc[i])
        # print(data)
        # print(newsDataFrame.head())
        documentColumn = df[df['NEWS_DATE'] == stockDataDf.iloc[i]['Date']].document
        string = ""
        for index, item in enumerate(documentColumn):
            string += " " + item
        data = {
            'Date': stockDataDf.iloc[i]['Date'],
            'document': string,
            'label': 0
        }
        newsDataFrame = newsDataFrame.append(data, ignore_index=True)

    pddf = pd.read_csv('./stockdatas/' + symbol + '_training.csv', parse_dates=True, index_col=0)
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

            string = ""
            for index, item in enumerate(sliceNewsDataFrame.document):
                string += " " + item
            data = {
                'document': string,
                'label': label
            }
            newsTrainData = newsTrainData.append(data, ignore_index=True)
            # study
            # https://stackoverflow.com/questions/17995328/changing-values-in-pandas-dataframe-does-not-work

    print("training labeling end")
    pddf = pd.read_csv('./stockdatas/' + symbol + '_testing.csv', parse_dates=True, index_col=0)
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
            string = ""
            for index, item in enumerate(sliceNewsDataFrame.document):
                string += " " + item
            data = {
                'document': string,
                'label': label
            }
            newsTestData = newsTestData.append(data, ignore_index=True)
    print("testing labeling end")
    print("countTrain : ", len(newsTrainData))
    print("countTest : ", len(newsTestData))

    newsTrainData = newsTrainData.sort_values(by=['label'])
    newsTestData = newsTestData.sort_values(by=['label'])

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

    # mecab = Mecab()
    #
    # train_data['tokenized'] = train_data['document'].apply(mecab.morphs)
    # train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
    # test_data['tokenized'] = test_data['document'].apply(mecab.morphs)
    # test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
    #
    # from matplotlib import font_manager, rc
    # font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    # rc('font', family=font_name)
    #
    #
    # down_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
    # up_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)
    # negative_word_count = Counter(down_words)
    # print(negative_word_count.most_common(20))
    # ## 상위 10개 단어 추출
    # top10_novel_bow = negative_word_count.most_common(10)
    # ## 10개 단어 시각화를 위한 기반 작업
    # n_groups = len(top10_novel_bow)
    # index = np.arange(n_groups)
    # ## 단어빈도수와 단어 리스트 자료형 준비
    # bow_vals = [x[1] for x in top10_novel_bow]
    # bow_words = [x[0] for x in top10_novel_bow]
    # ## 막대그래프 시각화
    # bar_width = 0.25
    # plt.bar(index, bow_vals, bar_width, color='b', label='Ocurrences')
    # plt.xticks(index, bow_words)
    # plt.tight_layout()
    # plt.show()
    #
    # return

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

    print('<종목코드', str_jong, '> 훈련용 빈샘플제거 후 개수', len(news_X_train))
    print('<종목코드', str_jong, '> 테스트용 빈샘플제거 후 개수', len(news_Y_train))

    max_len = 100

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
    # print(news_X_train[:3])
    # print("------------------------------------------------------------------------------------------")
    news_X_test = pad_sequences(news_X_test, maxlen=max_len)

    from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import confusion_matrix

    news_model = Sequential()
    news_model.add(Embedding(vocab_size, 128))
    # news_model.add(Bidirectional(LSTM(256, return_sequences = True)))
    news_model.add(Bidirectional(LSTM(256)))
    # news_model.add(LSTM(128))
    # news_model.add(Dense(2, activation='sigmoid'))
    news_model.add(Dense(128, activation="relu"))
    news_model.add(Dense(10, activation="relu"))

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    from tensorflow.keras.layers import concatenate
    combinedInput = concatenate([news_model.output, model.output])
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(10, activation="relu")(combinedInput)
    x = Dense(2, activation="softmax")(x)
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the house)
    model = Model(inputs=[news_model.input, model.input], outputs=x)

    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    opt = Adam(lr=1e-05)

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # train the model
    print("[INFO] training model...")
    news_X_train = np.asarray(news_X_train).astype(np.float32)
    news_Y_train = np.asarray(news_Y_train).astype(np.float32)
    news_X_test = np.asarray(news_X_test).astype(np.float32)
    news_Y_test = np.asarray(news_Y_test).astype(np.float32)

    # dataset_12 = tf.data.Dataset.from_tensor_slices(())
    # dataset_label = tf.data.Dataset.from_tensor_slices((Y_train))

    # dataset = tf.data.Dataset.zip((dataset_12,dataset_label)).shuffle(3, reshuffle_each_iteration=True).batch(2)

    es = EarlyStopping(monitor='loss', mode='max', verbose=1, patience=10)
    file_name = './{}epochs_{}batch_resnet+lstm_model_{}.h5'.format(epochs, batch_size, symbol)
    # file_name = r'c:\temp\file1'
    mc = ModelCheckpoint(file_name, monitor='loss', mode='min', verbose=1, save_best_only=True)

    # model.fit(x=[news_X_train, X_train], y=Y_train, epochs=epochs, callbacks=[es,mc])

    print("------------------------------------------------------------------------------------------")
    print("X_train size : ", X_train.shape)
    print("Y_train size : ", Y_train.shape)

    print("X_test size : ", X_test.shape)
    print("Y_test size : ", Y_test.shape)

    print("news_X_train size : ", news_X_train.shape)
    # print(news_X_train[0])
    # print(news_X_train[0].shape)
    print("news_X_test size : ", news_X_test.shape)
    print("news_X_test size : ", news_Y_train.shape)
    print("news_Y_test size : ", news_Y_test.shape)
    # make predictions on the testing data
    print("[INFO] predicting house prices...")
    loaded_model = load_model(file_name)
    preds = loaded_model.predict([news_X_test, X_test])
    print("------------------------------------------------------------------------------------------")
    print("preds : ", preds)
    print("Y_test : ", Y_test)

    y_preds = np.argmax(preds, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    # allTest = np.argmax([news_Y_test,Y_test], axis=1)

    print(accuracy_score(Y_test, y_preds))
    print("------------------------------------------------------------------------------------------")
    ################################## 뉴스 데이터 추출 및 병합끝 ##########################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    cm = confusion_matrix(Y_test, y_preds)
    report = classification_report(Y_test, y_preds)
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

    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    f_output.write('news+chart {}epochs_{}batch_lstm+resnet\n'.format(
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


if __name__ == "__main__":
    main()
