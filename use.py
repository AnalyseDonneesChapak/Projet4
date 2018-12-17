import random
import numpy as np
np.random.seed(123)  # for reproducibility
import matplotlib.pyplot as plt
import argparse
import shap


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.datasets import mnist
from os import listdir
from os.path import join, isfile
from numpy_vectors import load_data
from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import zero_one_loss
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb, rgb2gray, label2rgb  # since the code wants color images


class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)

class KerasMNIST:
    def __init__(self):
        self.model = self.get_model()

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
        score = loaded_model.evaluate(X_test, Y_test, verbose=0)

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
                  batch_size=32, epochs=10, verbose=0)

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
        i = np.expand_dims(image, axis=0)
        i = np.expand_dims(i, axis=4)
        r: np.ndarray = self.model.predict(i)
        r_list: list = r.tolist()
        # plt.plot(self.model.predict)
        return str(r_list[0].index(max(r_list[0])))

    def explain(self, image, save_to=None):


        # select a set of background examples to take an expectation over
        # i = np.expand_dims(image, axis=0)
        # i = np.expand_dims(i, axis=4)
        #
        # background = i
        # explain prediction on the model
        X_train, Y_train, X_test, Y_test = self.get_data()
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

        e = shap.DeepExplainer(self.model, background)

        image = image.reshape(1, 28, 28, 1)

        shap_values = e.shap_values(image)
        #plot the feature attributions
        plt.close('all')
        shap.image_plot(shap_values, -image, show=False)
        if save_to:
            plt.savefig(save_to)
        else:
            plt.show()

        return True


class KerasMNISTLimeExplainer(KerasMNIST):
    def __init__(self):
        super().__init__()
        # self.model = self.get_model()

    def get_data(self):
        from sklearn.model_selection import train_test_split
        mnist = fetch_mldata('MNIST original')
        # make each image color so lime_image works correctly
        X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
        y_vec = mnist.target.astype(np.uint8)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, train_size=0.55)

        return X_train, X_test, y_train, y_test

    def _get_from_file(self):
        raise FileNotFoundError()

    def _save_to_file(self, model):
        pass

    def _get_from_compile(self):
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import Normalizer

        class PipeStep(object):
            """
            Wrapper for turning functions into pipeline transforms (no-fitting)
            """

            def __init__(self, step_func):
                self._step_func = step_func

            def fit(self, *args):
                return self

            def transform(self, X):
                return self._step_func(X)

        makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
        flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

        simple_rf_pipeline = Pipeline([
            ('Make Gray', makegray_step),
            ('Flatten Image', flatten_step),
            # ('Normalize', Normalizer()),
            # ('PCA', PCA(16)),
            ('RF', RandomForestClassifier())
        ])

        X_train, X_test, y_train, y_test = self.get_data()

        simple_rf_pipeline.fit(X_train, y_train)

        return simple_rf_pipeline

    def predict(self, image):
        from lime import lime_image
        from lime.wrappers.scikit_image import SegmentationAlgorithm

        from matplotlib import pyplot as plt

        X_train, X_test, y_train, y_test = self.get_data()

        i = gray2rgb(image)
        i2 = X_test[0]

        r = 5


        explainer = lime_image.LimeImageExplainer(verbose=False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(i,
                                                 classifier_fn=self.model.predict_proba,
                                                 top_labels=10, hide_color=0, num_samples=10000,
                                                 segmentation_fn=segmenter)

        temp, mask = explanation.get_image_and_mask(r, positive_only=True, num_features=10, hide_rest=False,
                                                    min_weight=0.01)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
        ax1.set_title('Positive Regions for {}'.format(r))
        temp, mask = explanation.get_image_and_mask(r, positive_only=False, num_features=10, hide_rest=False,
                                                    min_weight=0.01)
        ax2.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
        ax2.set_title('Positive/Negative Regions for {}'.format(r))

        plt.show()

        # now show them for each class
        fig, m_axs = plt.subplots(2, 5, figsize=(12, 6))
        for i_, c_ax in enumerate(m_axs.flatten()):
            temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False,
                                                        min_weight=0.01)
            c_ax.imshow(label2rgb(mask, i, bg_label=0), interpolation='nearest')
            c_ax.set_title('Positive for {}\nActual {}'.format(i_, r))
            c_ax.axis('off')

        plt.show()

        return 5


class ScikitLearnMNIST:
    def __init__(self):
        self.model = self.get_model()
    def ESTIM(self):
        ESTIMATORS = {
            "dummy": DummyClassifier(),
            'CART': DecisionTreeClassifier(),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'Nystroem-SVM': make_pipeline(
                Nystroem(gamma=0.015, n_components=1000), SVC(C=100,kernel='linear',probability=True)),
            'SampledRBF-SVM': make_pipeline(
                RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)),
            'LogisticRegression-SAG': LogisticRegression(solver='sag', tol=1e-1,
                                                         C=1e4),
            'LogisticRegression-SAGA': LogisticRegression(solver='saga', tol=1e-1,
                                                          C=1e4),
            'MultilayerPerceptron': MLPClassifier(
                hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                solver='sgd', learning_rate_init=0.2, momentum=0.9, verbose=1,
                tol=1e-4, random_state=1),
            'MLP-adam': MLPClassifier(
                hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                solver='adam', learning_rate_init=0.001, verbose=1,
                tol=1e-4, random_state=1)
        }
        return ESTIMATORS
    def get_data(self,dtype=np.float32, order='F'):
        """Load the data, then cache and memmap the train/test split"""
        ######################################################################
        # Load dataset
        print("Loading dataset...")
        data = fetch_mldata('MNIST original')
        X = check_array(data['data'], dtype=dtype, order=order)
        y = data["target"]

        # Normalize features
        X = X / 255
        print("Creating train-test split...")
        n_train = 60000
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]

        return X_train, X_test, y_train, y_test

    def _save_to_file(self, model):
        # 11a. Serialize model to JSON
        joblib.dump(model,"scikit/model.joblib.pk1",)
        print("saved")
    def _get_from_file(self):
        # 1. Load json and create model
        model=joblib.load("scikit/model.joblib.pk1")
        return model
    # Create a classifier: a support vector classifier
    def _get_from_compile(self):
        #___-Chargement du calssifeirs et des parametres-___#
        ESTIMATORS=self.ESTIM()
        parser = argparse.ArgumentParser()
        parser.add_argument('--classifiers', nargs="+",
                            choices=ESTIMATORS, type=str,
                            default=['ExtraTrees', 'Nystroem-SVM'],
                            help="liste des classifiers de reference")
        parser.add_argument('--n-jobs', nargs="?", default=1, type=int,
                            help="Nombre de Number d'entraineur en cours de d'execution")
        parser.add_argument('--order', nargs="?", default="C", type=str,
                            choices=["F", "C"],
                            help="permet de choisir entre Fortran et ordre C data")
        parser.add_argument('--random-seed', nargs="?", default=0, type=int,
                            help="graine  choisis pour la génération aléatoire")
        args = vars(parser.parse_args())
        X_train, X_test, y_train, y_test = self.get_data(order=args["order"])
        error= {}
        #____________-Entrainement de l'I.A.-____________#
        for name in sorted(args["classifiers"]):
            print("Training %s ... " % name, end="")
            estimator = ESTIMATORS[name]
            estimator_params = estimator.get_params()

            estimator.set_params(**{p: args["random_seed"]
                                    for p in estimator_params
                                    if p.endswith("random_state")})

            if "n_jobs" in estimator_params:
                estimator.set_params(n_jobs=args["n_jobs"])

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            error[name] = zero_one_loss(y_test, y_pred)

        return estimator


    def get_model(self):
        try:
            model = self._get_from_file()
            #print("Loaded model from file")
        except FileNotFoundError as e:
            print(f"Couldn't load model from file with error {e}, compiling and saving model")
            model = self._get_from_compile()
            self._save_to_file(model)
        return model

    def predict(self, image):
        estimator= self.get_model()
        Image=image.reshape(1,784)
        Image=Image/255
        plt.imshow(image)
        plt.show()
        r = estimator.predict(Image)#► entrainer avec 10000 image en 784
        self.explainLIME(estimator,image,int(r))
        return str(int(r[0]))
    def explainLIME(self,model,image,estim):

        rgbim=np.zeros((28,28,3), 'uint8')
        rgbim[...,0]=image
        rgbim[...,1]=image
        rgbim[...,2]=image
        plt.plot(rgbim[0],rgbim[1])
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(rgbim, classifier_fn = model.predict_proba, top_labels=10,
                                                 hide_color=0, num_samples=10000, segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(estim, positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01)
        fig, ax1 = plt.subplots(1,1)
        ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
        ax1.set_title('positiv region for: {}'.format(estim))
        plt.show()


class RandomMNIST:
    def __init__(self):
        pass

    def get_data(self):
        pass

    def get_model(self) -> Sequential:
        pass

    def predict(self, image: np.array) -> str:
        return str(random.randint(0, 9))



if __name__ == '__main__':
    #e = KerasMNIST()
    #e.explain(load_data.load_image("dataset/training/8/0001.png"), )
    classifiers = [
        #KerasMNIST(),
        #ScikitLearnMNIST(),
        #RandomMNIST(),

        #KerasMNISTLimeExplainer()
    ]

    total_tests = 0
    answers = {c: 0 for c in classifiers}
    threads = []
    for i in range(10):
        FOLDER = f"dataset/testing/{i}/"
        for image_file in [f for f in listdir(FOLDER) if isfile(join(FOLDER, f))]:
            total_tests += 1
            #print(f"{i}/{image_file}")
            image = load_data.load_image(f"{FOLDER}/{image_file}")
            for classifier in classifiers:
                r = int(classifier.predict(image))
                answers[classifier] += int(r == i)
                #print("chiffre detecté: " + str(r))
                #print(f"{type(classifier).__name__} > good={r == i} (r={r}, i={i})")

                if r!=i and isinstance(classifier, KerasMNIST):
                    save_to = f'{i}_{image_file[:-4]}_{r}.png'
                    print(f"ERROR! Saving to {save_to}")

                    classifier.explain(image, save_to)

                    #thread = Thread(target=classifier.explain, args=(image, save_to,))
                    #thread.start()

                    #threads.append(thread)

                #print(f"{type(classifier).__name__} > good={r == i} (r={r}, i={i})")

    #for thread in threads:
    #    thread.join()

    for classifier, good in answers.items():
        print(
            f"{type(classifier).__name__} > Gave {good} good answers out of {total_tests} answers. ({good/total_tests*100} %)")
