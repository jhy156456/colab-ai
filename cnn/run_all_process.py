import subprocess
stock_code = "272210.KS"
day="5"
image_dimension = "100"
epochs = "10"
#epochs를 20으로 하면 오류가 나네..?
batch_size = "8"
try:
    #  read_csv :  stockdatas/BBNI.JK_testing.csv

    # print(f'python run_binary_preprocessing.py "BBNI.JK" "20" "50"')
    # subprocess.call(f'python run_binary_preprocessing.py  "BBNI.JK" "20" "50" ', shell=True)
    # print(f'python generatedata.py "dataset" "20_50/BBNI.JK" "dataset_BBNIJK_20_50" ')
    # subprocess.call(f'python generatedata.py "dataset" "20_50/BBNI.JK" "dataset_BBNIJK_20_50" ', shell=True)

    # print(f'python myDeepCNN.py "-i" "dataset/dataset_BBNIJK_20_50" "-e" "50" "-d" "50" "-b" "8" "-o" "outputresult.txt"')
    # subprocess.call(f'python myDeepCNN.py "-i" "dataset/dataset_BBNIJK_20_50" "-e" "50" "-d" "50" "-b" "8" "-o" "outputresult.txt"', shell=True)





    # print(f'python run_binary_preprocessing.py {stock_code} {day} {image_dimension}')
    # subprocess.call(f'python run_binary_preprocessing.py  {stock_code} {day} {image_dimension} ', shell=True)


    #원본 데이터 훼손 방지를 위해 별도로 분리한 파일을 학습하는 곳으로 복사한다.
    # print(f'python generatedata.py "dataset" "{day}_{image_dimension}/{stock_code}" "dataset_{stock_code}_{day}_{image_dimension}" ')
    # subprocess.call(f'python generatedata.py "dataset" "{day}_{image_dimension}/{stock_code}" "dataset_{stock_code}_{day}_{image_dimension}" ', shell=True)

    print(f'python myDeepCNN.py "-i" "dataset/dataset_{stock_code}_{day}_{image_dimension}" "-e" {epochs} "-d" {image_dimension} "-b" {batch_size} "-o" "outputresult.txt"')
    subprocess.call(f'python myDeepCNN.py "-i" "dataset/dataset_{stock_code}_{day}_{image_dimension}" "-e" {epochs} "-d" {image_dimension} "-b" {batch_size} "-o" "outputresult.txt"', shell=True)

except Exception as identifier:
    print(identifier)