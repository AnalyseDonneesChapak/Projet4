import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist

from numpy_vectors import load_data


class KerasMNIST:
    def __init__(self):
        X_train, Y_train, X_test, Y_test = self.get_data()

        self.model = self.get_model()
        # score = round(self.model.evaluate(X_test, Y_test, verbose=1)[1] * 100, 4)

        # print(f"Loaded model with score {score}%")

    def get_data(self):
        # 4. Load pre-shuffled MNIST data into train and test sets
        from keras.utils import np_utils

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # 5. Preprocess input data
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # 6. Preprocess class labels
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

        return X_train, Y_train, X_test, Y_test

    def _get_from_file(self):
        # 1. Load json and create model

        X_train, Y_train, X_test, Y_test = self.get_data()

        with open('keras/model.json', 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)

        # 2. Load weights into new model
        loaded_model.load_weights("keras/model.h5")

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(X_test, Y_test, verbose=1)

        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

        return loaded_model

    def _save_to_file(self, model):
        # 11a. Serialize model to JSON
        model_json = model.to_json()
        with open("keras/model.json", "w") as json_file:
            json_file.write(model_json)

        # 11b. Serialize weights to HDF5
        model.save_weights("keras/model.h5")

    def _get_from_compile(self):

        X_train, Y_train, X_test, Y_test = self.get_data()

        # 7. Define model architecture
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # 8. Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # 9. Fit model on training data
        model.fit(X_train, Y_train,
                  batch_size=32, epochs=10, verbose=1)

        # 10. Evaluate model on test data
        # score = model.evaluate(X_test, Y_test, verbose=1)

        return model

    def get_model(self) -> Sequential:
        try:
            model = self._get_from_file()
            print("Loaded model from file")
        except FileNotFoundError as e:
            print(f"Couldn't load model from file with error {e}, compiling and saving model")
            model = self._get_from_compile()
            self._save_to_file(model)

        return model

    def predict(self, image: np.array) -> str:
        #i = np.expand_dims(image, axis=0)
        #i = np.expand_dims(i, axis=4)
        #return self.model.predict(i)
        return ''


if __name__ == '__main__':
    k = KerasMNIST()
    i = load_data.load_image("dataset/testing/0/0001.png")
    r = k.predict(i)
    print(r)
    assert r == '0'

