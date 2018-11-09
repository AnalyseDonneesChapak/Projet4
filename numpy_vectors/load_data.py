import os
import numpy as np
from PIL import Image


DATASET_DIRECTORY = "dataset/training"

X = []
Y = []

def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


for i in range(10):
    folder = f"{DATASET_DIRECTORY}/{i}/"

    for image_file in os.listdir(folder):
        print(f"{folder}/{image_file}")
        X.append(load_image(f"{folder}/{image_file}"))
        Y.append(i)


X = np.array(X)  # Array de matrices des pixels en niveaux de gris
Y = np.array(Y)  # Array des "solutions" de l'index corrspondant dans X

assert len(X) == len(Y)

np.savez_compressed("TRAINING", X=X, Y=Y)

# Pour load, np.load("TRAINING")
