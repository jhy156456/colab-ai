import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.dates as mdates

str_jong = "272210.KS.xlsx"
symbol = "272210.KS"
df = pd.read_excel(str_jong, engine='openpyxl')
training_start_date = "2000-01-01"
training_end_date = "2021-05-31"

testing_start_date = "2021-06-01"
testing_end_date = "2021-12-31"
seq_len = 50

# str_jong = "{}".format(i).split('.')[0]
df["DOCUMENT"] = ""
df["label"] = 0
for i in range(len(df)):
    df["DOCUMENT"][i] = str(df.TITLE[i])+str(df.CODE[i])
print(df.NEWS_DATE)


pddf = pd.read_csv('../stockdatas/'+symbol+'_training.csv', parse_dates=True, index_col=0)
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

        starting = c["Close"].iloc[-2]
        endvalue = c["Close"].iloc[-1]
        tmp_rtn = endvalue / starting - 1
        if tmp_rtn > 0:
            # 상승
            label = 1
        else:
            # 하락
            label = 0
        df.loc[df['NEWS_DATE'] == c.iloc[i]["Date"]].label = label
        print(df.iloc[i])


