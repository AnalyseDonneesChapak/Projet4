import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))],0)
y_vec = mnist.target.astype(np.uint8)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.model_selection import train_test_split
import os,sys
from os import listdir
from os.path import isfile, join
from numpy_vectors import load_data
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


makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

simple_rf_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    #('PCA', PCA(16)),
    ('RF', RandomForestClassifier())
                              ])
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                    train_size=0.55)
simple_rf_pipeline.fit(X_train, y_train)
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
FOLDER = f"dataset/testing/5/"
for image_file in [f for f in listdir(FOLDER) if isfile(join(FOLDER, f))]:
    print(f"5/{image_file}")
    image = load_data.load_image(f"{FOLDER}/{image_file}")
    rgbim=np.zeros((28,28,3), 'uint8')
    rgbim[...,0]=image
    rgbim[...,1]=image
    rgbim[...,2]=image
    plt.plot(rgbim[0],rgbim[1])
    explainer = lime_image.LimeImageExplainer(verbose = False)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
    explanation = explainer.explain_instance(rgbim, classifier_fn = simple_rf_pipeline.predict_proba, top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
    temp, mask = explanation.get_image_and_mask(5, positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01)
    fig, ax1 = plt.subplots(1,1)
    ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    ax1.set_title('positiv region for: {}'.format(5))
    plt.show()