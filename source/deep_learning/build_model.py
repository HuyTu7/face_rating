from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import metrics
import logging

import numpy as np
import pandas as pd
np.random.seed(100)

import utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
import threading
import Queue

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                        level=logging.INFO)

class Dataset:
    def __init__(self, csv_path=None, dump=False):
        if not dump:
            self.raw_data = raw_data = pd.read_csv(csv_path) 
            raw_data.info()
            queue = Queue.Queue()
            for index, row in raw_data.iterrows():
                t = threading.Thread(target=utils.load_image,\
                args=(row['Attractiveness label'],row['Files'], queue,))
                t.daemon = True
                t.start()
            queue.join()
            item = queue.get()
            self.X = item['data']
            self.Y = item['y']
            while not queue.empty():
                item = queue.get()
                self.X = np.vstack((self.X, item['data']))
                self.Y = np.vstack((self.Y, item['y']))
           
            self.Y /= 5.0
            self.X.dump('./dataset/data_x.numpy')
            self.Y.dump('./dataset/data_y.numpy')
        else:
            self.X = np.load('./dataset/data_x.numpy')
            self.Y = np.load('./dataset/data_y.numpy')
        logging.info("shape of train data: %s"%str(self.X.shape)) 
        logging.info("Load data Done!")
    def getTrainTest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, \
                test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test
      
