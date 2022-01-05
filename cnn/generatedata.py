import os
from shutil import copyfile
import sys


def cre8outputdir(pathdir, targetdir):
    # create folder output
    if not os.path.exists("{}/{}".format(pathdir, targetdir)):
        os.mkdir("{}/{}".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train/0".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/0".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train/1".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/1".format(pathdir, targetdir))
    if not os.path.exists("{}/{}/train/2".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/2".format(pathdir, targetdir))
    if not os.path.exists("{}/{}/train/3".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/3".format(pathdir, targetdir))


    if not os.path.exists("{}/{}/test/0".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/0".format(pathdir, targetdir))
    if not os.path.exists("{}/{}/test/1".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/1".format(pathdir, targetdir))
    if not os.path.exists("{}/{}/test/2".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/2".format(pathdir, targetdir))
    if not os.path.exists("{}/{}/test/3".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/3".format(pathdir, targetdir))


pathdir = sys.argv[1]
origindir = sys.argv[2]
targetdir = sys.argv[3]

cre8outputdir(pathdir, targetdir)

counttest = 0
counttrain = 0
# 원본 데이터 훼손 방지를 위해 별도로 분리한 파일을 학습하는 곳으로 복사한다.
for root, dirs, files in os.walk("{}/{}".format(pathdir, origindir)):
    print("pathdir : " ,pathdir)
    print("origindir : ", origindir)
    for file in files:
        # print("file : ",file)

        tmp = root.replace('\\', '/')
        tmp_label = tmp.split('/')[-1]
        if 'test' in file:
            origin = "{}/{}".format(root, file)
            destination = "{}/{}/test/{}/{}".format(
                pathdir, targetdir, tmp_label, file)
            copyfile(origin, destination)
            counttest += 1
        elif 'train' in file:
            origin = "{}/{}".format(root, file)
            destination = "{}/{}/train/{}/{}".format(
                pathdir, targetdir, tmp_label, file)
            copyfile(origin, destination)
            counttrain += 1
        # if tmp_label == '0':
        #     if 'test' in file:
        #         origin = "{}/{}".format(root, file)
        #         destination = "{}/{}/test/0/{}".format(
        #             pathdir, targetdir, file)
        #         copyfile(origin, destination)
        #         counttest += 1
        #     elif 'train' in file:
        #         origin = "{}/{}".format(root, file)
        #         destination = "{}/{}/train/0/{}".format(
        #             pathdir, targetdir, file)
        #         copyfile(origin, destination)
        #         counttrain += 1
        # elif tmp_label == '1':
        #     if 'test' in file:
        #         origin = "{}/{}".format(root, file)
        #         destination = "{}/{}/test/1/{}".format(
        #             pathdir, targetdir, file)
        #         copyfile(origin, destination)
        #         counttest += 1
        #     elif 'train' in file:
        #         origin = "{}/{}".format(root, file)
        #         destination = "{}/{}/train/1/{}".format(
        #             pathdir, targetdir, file)
        #         copyfile(origin, destination)
        #         counttrain += 1

print("counttest : ", counttest)
print("counttrain : ", counttrain)
