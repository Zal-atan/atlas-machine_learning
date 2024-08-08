#!/usr/bin/env python3

import tensorflow as tf
Dataset = __import__('1-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(data.encode(pt, en))
for pt, en in data.data_valid.take(1):
    print(data.encode(pt, en))
