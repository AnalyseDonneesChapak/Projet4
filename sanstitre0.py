from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage.transform import resize
from numpy_vectors import load_data
from PIL import Image
import numpy as np
FOLDER = f"dataset/testing/3/"
for image_file in [f for f in listdir(FOLDER) if isfile(join(FOLDER, f))]:
    print(f"3/{image_file}")
    image = load_data.load_image(f"{FOLDER}/{image_file}")
    Image=Image=image.reshape(1,[28,28]) 
    plt.imshow(Image)
    plt.show()
    