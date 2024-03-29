from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from collections import defaultdict
import numpy as np
# import scipy.misc
from matplotlib.pyplot import imread


def dataset(base_dir, n):
    print("base_dir : {}, n : {}".format(base_dir, n))
    # base_dir : "dataset/dataset_272210.KS_{day}_{image_dimension}"
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        print('root :', root)
        print('subdirs :', subdirs)
        print('files :', files)

        # for filename in subdirs:
        for filename in files:
            file_path = os.path.join(root, filename)
            # print(' ok :', file_path)

            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("\\")
            # label = suffix.split("\\")[0]
            label = suffix[1]
            d[label].append(file_path)

    tags = sorted(d.keys())
    print("tags : ", tags)
    print("classes : {}".format(tags))

    X = []
    y = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        count = 0
        for filename in filenames:
            # print('filename :: ', filename)
            # img = scipy.misc.imread(filename) # 없어짐.
            img = imread(filename)
            height, width, chan = img.shape
            # assert chan == 4
            assert chan == 3
            X.append(img)
            y.append(class_index)
            count += 1
        print("class_index : ",class_index, " / count : " , count)

    X = np.array(X).astype(np.float32)
    y = np.array(y)

    return X, y, tags
