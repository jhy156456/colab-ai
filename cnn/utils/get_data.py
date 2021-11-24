import argparse
import arrow
import pandas as pd
import datetime as dt
from pandas_datareader import data, wb
import os
import yfinance as yf
import time
# fixed pandas_datareader can't download from yahoo finance
yf.pdr_override()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sd', '--start_date', type=str,
                        default='1990-01-01', help='Start date parameter value - format YYYY-MM-DD')
    parser.add_argument('-ed', '--end_date', type=str,
                        default=arrow.now().format('YYYY-MM-DD'), help='End date parameter - format YYYY-MM-DD')
    parser.add_argument('-t', '--ticker', nargs='+',
                        help='<Required> Set flag', required=True)
    parser.add_argument(
        '-s', '--source', help='<Required> set source', required=True)
    parser.add_argument('-a', '--attempt',
                        help='set max attempt to download', default=10)
    parser.add_argument(
        '-e', '--exist', help='check exist stock history file', default=False)
    parser.add_argument('-p', '--prefix', help='add prefix in output name')
    args = parser.parse_args()
    # # fetch all data

    prefix_name = ""
    # make sure output folder is exist
    if not os.path.isdir("../stockdatas"):
        os.mkdir("../stockdatas")
    if len(args.prefix) > 1:
        prefix_name = args.prefix
    if args.source == "tiingo":
        for ticker in set(args.ticker):
            fetch_tiingo_data(ticker, args.start_date, args.end_date,
                              "../stockdatas/{}_{}.csv".format(ticker, prefix_name))
    elif args.source == "yahoo":
        for ticker in set(args.ticker):
            # fetch_yahoo_data(ticker, args.start_date, args.end_date,
            #                  "../stockdatas/{}_{}.csv".format(ticker, prefix_name), args.attempt, args.exist)
            fetch_yahoo_data(ticker, args.start_date, args.end_date,
                             "./stockdatas/{}_{}.csv".format(ticker, prefix_name), args.attempt, args.exist,args.prefix)


def fetch_tiingo_data(ticker, start_date, end_date, fname):
    url = "https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={token}"
    token = "ca5a6f47a99ae61051e4de63b26f727b1709a01d"
    data = pd.read_json(url.format(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        token=token
    ))
    data.to_csv(fname, columns=["date", "open", "close",
                                "high", "low", "volume", "adjClose"], index=False)


def fetch_yahoo_data(ticker, start_date, end_date, fname, max_attempt, check_exist,prefix):
    # if (os.path.exists(fname) == True) and check_exist:
    #     print("file exist")
    # else:
    #     # remove exist file
    #     if os.path.exists(fname):
    #         os.remove(fname)
    #     for attempt in range(max_attempt):
    #         time.sleep(2)
    #         try:
    #             dat = data.get_data_yahoo(''.join("{}".format(
    #                 ticker)),  start=start_date, end=end_date)
    #             print("fname : " + fname)
    #             dat.to_csv(fname)
    #         except Exception as e:
    #             if attempt < max_attempt - 1:
    #                 print('Attempt {}: {}'.format(attempt + 1, str(e)))
    #             else:
    #                 raise
    #         else:
    #             break


    dat = pd.read_csv(fname)
    print(dat.head())
    ############ 뉴스영역 시작 ############
    news_filename = "./stockdatas/{}_news.xlsx".format(fname[13:22])
    news_df = pd.read_excel(news_filename, engine='openpyxl')
    news_df = news_df.iloc[::-1].set_index(news_df.index)
    # study : 열 합치기
    # https://www.delftstack.com/ko/howto/python-pandas/how-to-combine-two-columns-of-text-in-dataframe-in-pandas/
    news_df["Content"] = news_df["Title"].map(str) + " " + news_df["Content"].map(str)




    # news_df = news_df.reindex(index=news_df.index[::-1])
    # news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date

    # date에 시간이 있어서 시간부분을 삭제한다.
    # study
    # type : <class 'datetime.date'>
    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.strftime('%Y-%m-%d')



    # start_indexs = news_df[news_df['Date'] == start_date].index.tolist()
    # if len(start_indexs) == 0:
    #     print(len(news_df))
    #     start_indexs.append(0)
    #
    # end_indexs = news_df[news_df['Date'] == end_date].index.tolist()
    # if len(end_indexs) == 0:
    #     print(len(news_df))
    #     end_indexs.append(len(news_df)-1)
    #
    # news_prefix_df = news_df.iloc[start_indexs[0]:end_indexs[len(end_indexs)-1],:]

    news_prefix_df = pd.DataFrame(columns = ['Date' , 'Content'])
    for i in range(0, len(dat)):
        add_df = pd.DataFrame(news_df[news_df['Date']==dat.iloc[i]['Date']],columns=['Date','Content'])
        if len(add_df) ==0:
            add_df = pd.DataFrame([ (dat.iloc[i]['Date'],"")],columns=['Date','Content'])
        news_prefix_df = news_prefix_df.append(add_df)


    news_fname_prefix = "./stockdatas/{}_news_{}.csv".format(fname[13:22],prefix)
    if os.path.exists(news_fname_prefix):
        os.remove(news_fname_prefix)

    try:
        print("fname : " + news_fname_prefix)
        news_prefix_df.to_csv(news_fname_prefix,index=False , encoding="utf-8-sig")
    except Exception as e:
        print("save error")
    ############ 뉴스영역 끝 ############

if __name__ == '__main__':
    main()
