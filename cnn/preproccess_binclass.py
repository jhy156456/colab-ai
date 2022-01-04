import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
from shutil import move
from pathlib import Path

# https://github.com/matplotlib/mpl_finance
# from mpl_finance import candlestick2_ochl, volume_overlay
from mplfinance.original_flavor import candlestick2_ochl, volume_overlay


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


def removeOutput(finput):
    if (Path(finput)).is_file():
        os.remove(finput)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='a csv file of stock data', required=True)
    parser.add_argument('-l', '--seq_len',
                        help='num of sequence length', default=20)
    parser.add_argument('-lf', '--label_file',
                        help='a label_file')
    parser.add_argument('-d', '--dimension',
                        help='a dimension value', type=int, default=48)
    parser.add_argument('-t', '--dataset_type',
                        help='training or testing datasets')
    parser.add_argument('-m', '--mode',
                        help='mode of preprocessing data', required=True)
    parser.add_argument('-v', '--use_volume',
                        help='combine with volume.', default=False)
    args = parser.parse_args()
    if args.mode == 'ohlc2cs':
        ohlc2cs(args.input, args.seq_len, args.dataset_type,
                args.dimension, args.use_volume)
    if args.mode == 'createLabel':
        createLabel(args.input, args.seq_len)
    if args.mode == 'img2dt':
        image2dataset(args.input, args.label_file)
    if args.mode == 'countImg':
        countImage(args.input)


# 학습 데이터를 종목 코드별로 폴더로 옮기는 작업
def image2dataset(input, label_file):
    print("*****image2dataset****")
    # input : dataset/{windows_length}_{dimension}/{symbol}/testing
    # -lable_file :  ./label/{symbol}_testing_label_{windows_length}.txt'
    label_dict = {}
    # print("label_file :", label_file)
    with open(label_file) as f:
        for line in f:
            (key, val) = line.split(',')
            label_dict[key] = val.rstrip()
    # print(label_dict)
    path = "{}/{}".format(os.getcwd(), input)
    # path : C:\Users\korea\Desktop\jhy\dd\colab-ai\cnn/dataset/5_100/272210.KS/training
    # 처음 생성하는게 아니라 두번째 중복으로 생성하는 것이라면
    # filename에 기존에 생성되어있었던 classes파일이 들어가 있을것이고,
    # split("_")하면 길이가 1인 배열이 반환되므로
    # list index out of range 에러가 날것임, filename에서 classes는 종목코드가 아니므로 가장 마지막 인덱스에 위치해있으므로
    # 상관없다. 결론은 이 에러는 무시해도 된다.
    for filename in os.listdir(path):
        if filename != '':
            for k, v in label_dict.items():
                # print(filename)
                splitname = filename.split("_")
                # print(splitname)
                f, e = os.path.splitext(filename)
                # print("[DEBUG] - {}".format(splitname))
                newname = "{}_{}".format(splitname[0], splitname[1])
                if newname == k:
                    # print("{} same with {} with v {}".format(filename, k, v))
                    new_name = "{}{}.png".format(v, f)

                    os.rename("{}/{}".format(path, filename),
                              "{}/{}".format(path, new_name))
                    break

    folders = ['3','2','1', '0']
    for folder in folders:
        if not os.path.exists("{}/classes/{}".format(path, folder)):
            os.makedirs("{}/classes/{}".format(path, folder))

    for filename in os.listdir(path):
        if filename != '' and filename != 'classes':
            # print(filename[:1])
            ### 여기에 for k,v in label_dict.items() 돌면서
            f, e = os.path.splitext(filename)
            if label_dict[f] == "1":
                move("{}/{}".format(path, filename),
                     "{}/classes/1/{}".format(path, filename))
            elif label_dict[f] == "0":
                move("{}/{}".format(path, filename),
                     "{}/classes/0/{}".format(path, filename))
            elif label_dict[f] == "2":
                move("{}/{}".format(path, filename),
                     "{}/classes/2/{}".format(path, filename))
            elif label_dict[f] == "3":
                move("{}/{}".format(path, filename),
                     "{}/classes/3/{}".format(path, filename))

    print('Done')


"""
차트 이미지데이터에 레이블을 달고
뉴스데이터에는 해당 날짜의 익일 종가에대한 레이블을 달아준다.
0 : 3% 이상 상승 
1 : 0~3% 상승
2 : 0~-3% 하락
3 : -3%이하 하락
"""
def createLabel(fname, seq_len):
    # python preprocess.py -m createLabel -l 20 -i stockdatas/EWT_training5.csv
    print("*****createLabel****")
    # remove existing label file
    filename = fname.split('/')

    stock_code = filename[1][:9]
    prefix = filename[1][10:].split('.')[0]

    news_filename = "./" + filename[0] + "/" + stock_code + "_news_" + prefix + ".csv"

    # print("{} - {}".format(filename[0], filename[1][:-4]))
    removeOutput("./label/{}_label_{}.txt".format(filename[1][:-4], seq_len))
    stockdata_df = pd.read_csv(fname, parse_dates=True, index_col=0)
    stockdata_df.fillna(0)
    stockdata_df.reset_index(inplace=True)
    # stockdata_df['Date'] = stockdata_df['Date'].map(mdates.date2num)



    # newsdate_df = pd.read_csv(news_filename, parse_dates=True)
    # newsdate_df['label'] = -1

    print(seq_len)
    for i in range(0, len(stockdata_df)):
        # 수정!!  int(seq_len)+1 ->  int(seq_len)
        # 1일~seq_len+1 일치 데이터프레임 획득
        c = stockdata_df.iloc[i:i + int(seq_len) + 1, :]
        starting = 0
        endvalue = 0
        label = ""
        if len(c) == int(seq_len) + 1:
            # study : iloc[-1] : # 마지막 행만

            # seq_len +1 일치 시초가, 종가
            # starting = c["Open"].iloc[-1]
            starting = c["Close"].iloc[-2]
            endvalue = c["Close"].iloc[-1]
            tmp_value = (endvalue - starting) / starting * 100
            tmp_rtn = endvalue / starting - 1

            # if tmp_rtn > 0:
            #     label = 1
            # else :
            #     label = 0


            if 3.0 <= tmp_value:
                # 상승
                label = 0
            elif 0.0 <= tmp_value <= 3.0:
                # 하락
                label = 1
            elif 0.0 > tmp_value >= -3.0:
                label = 2
            else:
                label = 3



            with open("./label/{}_label_{}.txt".format(filename[1][:-4], seq_len), 'a') as the_file:
                the_file.write("{}-{},{}".format(filename[1][:-4], i, label))
                the_file.write("\n")
        # else :
        #     print(c)

        ############ 뉴스영역 시작 ############
    #     if len(stockdata_df) - 1 != i:
    #         starting = stockdata_df["Close"].iloc[i]
    #         endvalue = stockdata_df["Close"].iloc[i + 1]
    #         tmp_value = (endvalue - starting) / starting * 100
    #         tmp_rtn = endvalue / starting - 1
    #
    #         # if tmp_rtn > 0:
    #         #     label = 1
    #         # else:
    #         #     label = 0
    #
    #
    #         if 3.0 <= tmp_value:
    #             # 상승
    #             label = 0
    #         elif 0.0 <= tmp_value <= 3.0:
    #             # 하락
    #             label = 1
    #         elif 0.0 > tmp_value >= -3.0:
    #             label = 2
    #         else:
    #             label = 3
    #
    #
    #         # study
    #         # 일치하는 열들 값 변경하기
    #         # https://pongdangstory.tistory.com/518
    #         newsdate_df.loc[
    #             (newsdate_df["Date"] == stockdata_df["Date"].iloc[i + 1].strftime('%Y-%m-%d')), 'label'] = label
    #
    # if os.path.exists(news_filename):
    #     os.remove(news_filename)
    # try:
    #     print("fname : " + news_filename)
    #     newsdate_df.to_csv(news_filename, index=False, encoding="utf-8-sig")
    # except Exception as e:
    #     print("save error")
    ############ 뉴스영역 끝 ############
    print("Create label finished.")


def countImage(input):
    num_file = sum([len(files) for r, d, files in os.walk(input)])
    num_dir = sum([len(d) for r, d, files in os.walk(input)])
    print("num of files : {}\nnum of dir : {}".format(num_file, num_dir))


import matplotlib.gridspec as gridspec
import numpy as np


def ohlc2cs(fname, seq_len, dataset_type, dimension, use_volume):
    # python preprocess.py -m ohlc2cs -l 20 -i stockdatas/EWT_testing.csv -t testing
    print("Converting ohlc to candlestick")
    symbol = fname.split('_')[0]
    symbol = symbol.split('/')[1]
    print(symbol)
    path = "{}".format(os.getcwd())
    # print(path)
    if not os.path.exists("{}/dataset/{}_{}/{}/{}".format(path, seq_len, dimension, symbol, dataset_type)):
        os.makedirs("{}/dataset/{}_{}/{}/{}".format(path, seq_len, dimension, symbol, dataset_type))
    print("fname : " + fname);
    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    df.fillna(0)
    plt.style.use('dark_background')
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)
    # gs = gridspec.GridSpec(nrows=2,ncols=1,height_ratios=[3,1])
    # for i in range(0, len(df)):
    for i in range(0, len(df) - int(seq_len)):
        # ohlc+volume
        # 수정!!  int(seq_len)-1
        c = df.iloc[i:i + int(seq_len), :]
        if len(c) == int(seq_len):
            my_dpi = 96
            value = dimension / my_dpi
            # http://blog.quantylab.com/candlestick.html
            fig = plt.figure(figsize=(value,
                                      value), dpi=my_dpi)
            # fig.set_facecolor('w')
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            axes = []
            axes.append(plt.subplot(gs[0]))
            axes.append(plt.subplot(gs[1], sharex=axes[0]))
            # axes[0].get_xaxis().set_visible(False)

            x = np.arange(len(c.index))
            ohlc = c[['Open', 'Close', 'High', 'Low']].astype(int).values
            dohlc = np.hstack((np.reshape(x, (-1, 1)), ohlc))

            # 봉차트
            candlestick2_ochl(axes[0], c['Open'], c['Close'], c['High'], c['Low'], width=0.5, colorup='r',
                              colordown='b')
            axes[0].grid(False)
            axes[0].set_xticklabels([])
            axes[0].set_yticklabels([])
            axes[0].xaxis.set_visible(False)
            axes[0].yaxis.set_visible(False)
            axes[0].axis('off')
            # 거래량 차트
            axes[1].bar(x, c['Volume'], color='w', width=0.6, align='center')

            # volume_overlay(axes[1], c['Open'], c['Close'], c['Volume'],colorup='r', colordown='b',  width=0.6)

            axes[1].grid(False)
            axes[1].set_xticklabels([])
            axes[1].set_yticklabels([])
            axes[1].xaxis.set_visible(False)
            axes[1].yaxis.set_visible(False)
            axes[1].axis('off')

            plt.tight_layout()
            pngfile = 'dataset/{}_{}/{}/{}/{}-{}.png'.format(
                seq_len, dimension, symbol, dataset_type, fname[11:-4], i)
            fig.savefig(pngfile, pad_inches=0, transparent=False)
            plt.close(fig)
            # Alpha 채널 없애기 위한.
            from PIL import Image
            img = Image.open(pngfile)
            img = img.convert('RGB')
            img.save(pngfile)


print("Converting olhc to candlestik finished.")

# def ohlc2cs(fname, seq_len, dataset_type, dimension, use_volume):
#     # python preprocess.py -m ohlc2cs -l 20 -i stockdatas/EWT_testing.csv -t testing
#     print("Converting ohlc to candlestick")
#     symbol = fname.split('_')[0]
#     symbol = symbol.split('/')[1]
#     print(symbol)
#     path = "{}".format(os.getcwd())
#     # print(path)
#     if not os.path.exists("{}/dataset/{}_{}/{}/{}".format(path, seq_len, dimension, symbol, dataset_type)):
#         os.makedirs("{}/dataset/{}_{}/{}/{}".format(path, seq_len, dimension, symbol, dataset_type))
#
#     df = pd.read_csv(fname, parse_dates=True, index_col=0)
#     df.fillna(0)
#     plt.style.use('dark_background')
#     df.reset_index(inplace=True)
#     df['Date'] = df['Date'].map(mdates.date2num)
#     # gs = gridspec.GridSpec(nrows=2,ncols=1,height_ratios=[3,1])
#     # for i in range(0, len(df)):
#     for i in range(0, len(df) - int(seq_len)):
#         # ohlc+volume
#         # 수정!!  int(seq_len)-1
#         c = df.iloc[i:i + int(seq_len), :]
#         if len(c) == int(seq_len):
#             my_dpi = 96
#             value = dimension / my_dpi
#
#             fig = plt.figure(figsize=(value,
#                                       value), dpi=my_dpi)
#             # ax1 = fig.add_subplot(1, 1, 1)
#             # ax1 = fig.subplot2_grid(gs[0])
#             top_axes = plt.subplot2grid((5, 5), (0, 0), rowspan=3, colspan=5)
#             bottom_axes = plt.subplot2grid((5, 5), (3, 0), rowspan=2, colspan=5, sharex=top_axes)
#
#             candlestick2_ochl(top_axes, c['Open'], c['Close'], c['High'], c['Low'],
#                               width=1, colorup='r', colordown='b')
#             top_axes.grid(False)
#             top_axes.set_xticklabels([])
#             top_axes.set_yticklabels([])
#             top_axes.xaxis.set_visible(False)
#             top_axes.yaxis.set_visible(False)
#             top_axes.axis('off')
#
#             # create the second axis for the volume bar-plot
#             # Add a seconds axis for the volume overlay
#             if use_volume:
#                 # ax2 = ax1.twinx()
#
#                 # ax2 = fig.add_subplot(gs[1])
#                 # ax2 = ax1.twinx()
#
#
#
#                 # Plot the volume overlay
#                 bc = volume_overlay(bottom_axes, c['Open'], c['Close'], c['Volume'],
#                                     colorup='r', colordown='b', alpha=0.5, width=1)
#                 bottom_axes.add_collection(bc)
#                 bottom_axes.grid(False)
#                 bottom_axes.set_xticklabels([])
#                 bottom_axes.set_yticklabels([])
#                 bottom_axes.xaxis.set_visible(False)
#                 bottom_axes.yaxis.set_visible(False)
#                 bottom_axes.axis('off')
#             pngfile = 'dataset/{}_{}/{}/{}/{}-{}.png'.format(
#                 seq_len, dimension, symbol, dataset_type, fname[11:-4], i)
#             fig.savefig(pngfile, pad_inches=0, transparent=False)
#             plt.close(fig)
#
#             # Alpha 채널 없애기 위한.
#             from PIL import Image
#             img = Image.open(pngfile)
#             img = img.convert('RGB')
#             img.save(pngfile)
#
#         # normal length - end
#
#     print("Converting olhc to candlestik finished.")


if __name__ == '__main__':
    main()
