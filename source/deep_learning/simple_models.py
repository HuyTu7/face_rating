from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
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
    def __init__(self, csv_path, dump=False):
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
            self.X.dump('data_x.numpy')
            self.Y.dump('data_y.numpy')
        else:
            self.X = np.load('data_x.numpy')
            self.Y = np.load('data_y.numpy')
        logging.info("shape of train data: %s"%str(self.X.shape)) 
        logging.info("Load data Done!")
    def getTrainTest(self):
        pass
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, \
                test_size=0.33, random_state=42)
        return X_train, y_train, X_test, y_test
class BeautyModel:
    def __init__(self):
        self.model = model = Sequential()

        # model.add(Conv2D(50, (5, 5), input_shape=(227, 227, 3), activation='relu',\
                # kernel_constraint=maxnorm, padding='same'))
        model.add(Conv2D(50, (5, 5), input_shape=(227, 227, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(100, (4, 4), activation='relu', padding='same'))        
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        model.summary()

        self.datagen = ImageDataGenerator(
                rescale=1./255,
                )

    def train(self, train_X, train_Y, test_X, test_Y):
        #epochs, batch_size
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_datagen.fit(train_X)
        test_datagen.fit(test_X)
        steps = len(train_X)/32

        logging.info("Model is training....")
        self.model.fit_generator(
            train_datagen.flow(train_X, train_Y),
            steps_per_epoch=steps,
            epochs=10,
            validation_data=test_datagen.flow(test_X, test_Y),
            validation_steps=steps
        )
        # save the trained model
        # serialize model to JSON
        model_json = self.model.to_json()
      
        with open("simple_beauty.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("simple_beauty.h5")

        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps = steps)
        print "%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100)
    def predict(self):
        pass

if __name__ == '__main__':
   
    data = Dataset('./SCUT_FBP(227,227).csv', True)
    model = BeautyModel()

    train_x, train_y, test_x, test_y = data.getTrainTest()
    model.train(train_x, train_y, test_x, test_y)
    """example data to test the model is working"""
