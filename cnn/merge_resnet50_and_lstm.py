import os
import logging

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
from keras.layers import Input, Dense, Conv2D, ZeroPadding2D, Flatten, Activation, add,MaxPooling2D,AveragePooling2D
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
from fetch_newsdata_and_construct_model import fetch_newsdate_and_construct_model




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





    fetch_newsdate_and_construct_model(seq_len=30,epochs=epochs,batch_size=batch_size,symbol="055550.KS")


    return
    ################################# concate~fit ##################################
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
    # opt = Adam(lr=1e-3, decay=1e-3 / 200)
    opt = Adam(lr=1e-05)
    # opt = tf.optimizers.RMSprop(lr=0.001)

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # train the model
    print("[INFO] training model...")
    news_X_train = np.asarray(news_X_train).astype(np.float32)
    news_Y_train = np.asarray(news_Y_train).astype(np.float32)
    news_X_test = np.asarray(news_X_test).astype(np.float32)
    news_Y_test = np.asarray(news_Y_test).astype(np.float32)

    # dataset_12 = tf.data.Dataset.from_tensor_slices(())
    # dataset_label = tf.data.Dataset.from_tensor_slices((Y_train))

    # dataset = tf.data.Dataset.zip((dataset_12,dataset_label)).shuffle(3, reshuffle_each_iteration=True).batch(2)

    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=4)
    file_name = './{}epochs_{}batch_resnet+lstm_model_{}.h5'.format(epochs, batch_size, symbol)
    # file_name = r'c:\temp\file1'
    mc = ModelCheckpoint(file_name, monitor='loss', mode='min', verbose=1, save_best_only=True)

    model.fit(x=[news_X_train, X_train], y=Y_train, epochs=epochs, callbacks=[es,mc])
    ################################# concate~fit ##################################











    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
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


if __name__ == "__main__":
    main()
