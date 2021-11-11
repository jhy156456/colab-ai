import os

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)

import math
import keras

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.models import Model
from keras.utils import np_utils
from tensorflow.keras.optimizers import *
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
# from utils.dataset import dataset as dataset
from utils.dataset import dataset
import argparse

import time
from datetime import timedelta
import pandas as pd

def build_dataset(data_directory, img_width):
    # X, y, tags = dataset.dataset(data_directory, int(img_width))
    # data_directory : dataset/dataset_272210.KS_{day}_{image_dimension}
    X, y, tags = dataset(data_directory, int(img_width))
    print(len(tags))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    print(y)
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes


def build_model(SHAPE, nb_classes, bn_axis, seed=None):
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    # Step 1
    x = Conv2D(32, 3, 3, kernel_initializer='glorot_uniform',
               padding='same', activation='relu')(input_layer)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Step 1
    x = Conv2D(48, 3, 3, kernel_initializer='glorot_uniform', padding='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Step 1
    x = Conv2D(64, 3, 3, kernel_initializer='glorot_uniform', padding='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Step 1
    x = Conv2D(96, 3, 3, kernel_initializer='glorot_uniform', padding='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    # Step 3 - Flattening
    x = Flatten()(x)

    # Step 4 - Full connection

    x = Dense(256, activation='relu')(x)
    # Dropout
    x = Dropout(0.5)(x)

    x = Dense(2, activation='softmax')(x)

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

    # channel = 4
    # bn_axis = 4 if K.image_dim_ordering() == 'tf' else 1

    data_directory = args.input

    print("loading dataset")
    # data_directory : dataset/dataset_272210.KS_{day}_{image_dimension}
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), args.dimension)
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))

    model = build_model(SHAPE, nb_classes, bn_axis)

    model.compile(optimizer=Adam(learning_rate=1.0e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=10)
    # mc = ModelCheckpoint('best_model_samsung.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    file_name = '{}epochs_{}batch_resnet50_model_{}.h5'.format(epochs, batch_size, data_directory.replace("/", "_"))
    mc = ModelCheckpoint(file_name, monitor='loss', mode='min', verbose=1, save_best_only=True)
    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,callbacks=[es, mc])


    loaded_model = load_model(file_name)
    predicted = loaded_model.predict(X_test)
    # del model  # deletes the existing model
    # predicted = model.predict(X_test)

    print("predicted.shape : ",predicted.shape)
    print("Y_test.shape : ", Y_test.shape)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    print("------------------------------------------------------------------------------------------")
    print(y_pred)
    print(Y_test)
    print(accuracy_score(Y_test, y_pred))
    print("------------------------------------------------------------------------------------------")
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
    f_output.write('{}epochs_{}batch_cnn\n'.format(
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

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # Plot a confusion matrix.
    # cm is the confusion matrix, names are the names of the classes.
    def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Plot an ROC. pred - the predictions, y - the expected output.
    def plot_roc(pred, y):
        fpr, tpr, _ = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('ROC AUC.png')
        plt.show()

    plot_roc(y_pred, Y_test)


if __name__ == "__main__":
    main()
