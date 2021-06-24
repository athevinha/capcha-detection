# Imports
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil
import glob
import os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = './DATA_TRAIN/'  # DIR containing images
targetdir = "./TEST_3/"  # DIR to copy clustered images to
number_clusters = 10

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    try:
        print("    Status: %s / %s" % (i, len(filelist)), end="\r")
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
    except:
        continue

# Clustering
kmeans = KMeans(n_clusters=number_clusters,
                random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    try:
        print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
        shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".png")
    except:
        continue
