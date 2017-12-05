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
      
class BeautyModel:
    def __init__(self, load=False):
        if not load:
            self.model = model = Sequential()

            # model.add(Conv2D(50, (5, 5), input_shape=(227, 227, 3), activation='relu',\
                    # kernel_constraint=maxnorm, padding='same'))
            model.add(Conv2D(50, (5, 5), input_shape=(227, 227, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Conv2D(100, (5, 5), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(150, (4, 4), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(200, (4, 4), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(250, (4, 4), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(300, (2, 2), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(500, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(Dropout(0.2))
            model.add(Dense(5, activation='softmax', kernel_constraint=maxnorm(3)))
            model.add(Dense(1, activation='sigmoid'))
        else:
            from keras.models import model_from_json 
            json_file = open('./models/beauty.json')
            loaded_model_info = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_info)
            #load weight
            self.model.load_weights('./models/beauty.h5')
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        
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
        steps_train = len(train_X)/32
        steps_test = len(test_X)/32

        logging.info("Train data: ")
        logging.info("  train shape:  " + str(train_X.shape))
        logging.info("  test shape:  " + str(test_X.shape))
        logging.info("Model is training....")
        self.model.fit_generator(
            train_datagen.flow(train_X, train_Y, shuffle=True),
            steps_per_epoch=steps_train,
            epochs=10,
            validation_data=test_datagen.flow(test_X, test_Y, shuffle=True),
            validation_steps=steps_test
        )
        # save the trained model
        # serialize model to JSON
        model_json = self.model.to_json()

        with open("./models/beauty.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("./models/beauty.h5")

        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps = steps_test)
        print scores
        print "%s: %.2f%%" % (self.model.metrics_names[1], scores[1])

    def getEvaluate(self, test_X, test_Y):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen.fit(test_X)
        steps = len(test_X)/32
        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps=steps)
        print scores
        print "%s: %.2f%%" % (self.model.metrics_names[1], scores[1])

    def predict(self):
        pass

if __name__ == '__main__':
   
    data = Dataset(dump=True)
    model = BeautyModel()

    train_x, train_y, test_x, test_y = data.getTrainTest()
    model.train(train_x, train_y, test_x, test_y)
    # model.getEvaluate(test_x, test_y)
    """example data to test the model is working"""
