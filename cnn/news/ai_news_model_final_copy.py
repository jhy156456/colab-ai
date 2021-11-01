import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
str_jong = "272210.KS.xlsx"
df = pd.read_excel(str_jong, engine='openpyxl')
training_start_date = "2000-01-01"
training_end_date = "2021-05-31"

testing_start_date = "2021-06-01"
testing_end_date = "2021-12-31"


# str_jong = "{}".format(i).split('.')[0]
df["DOCUMENT"] = ""
df["label"] = 0
for i in range(len(df)):
    df["DOCUMENT"][i] = str(df.TITLE[i])+str(df.CODE[i])
print(df.NEWS_DATE)
