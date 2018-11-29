import random
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from numpy_vectors import load_data
from skimage.transform import resize
digits = datasets.load_digits() 
classifier = svm.SVC(gamma=0.001)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
print(classifier)
def Scikitlearn(img):
    print(img.size)
    predict=classifier.predict(img)
    print(predict)
    return predict

total_tests = 0
for i in range(10):
    FOLDER = f"dataset/testing/{i}/"
    print(i)
    for image_file in [f for f in listdir(FOLDER) if isfile(join(FOLDER, f))]:
        total_tests += 1
        print(f"{i}/{image_file}")
        image = load_data.load_image(f"{FOLDER}/{image_file}")
        r = int(Scikitlearn(image))
        answers[classifier] += int(r == i)

        print(f"{type(classifier).__name__} > good={r == i} (r={r}, i={i})")
for classifier, good in answers.items():
    print(f"{type(classifier).__name__} > Gave {good} good answers out of {total_tests} answers. ({good/total_tests*100} %)")
