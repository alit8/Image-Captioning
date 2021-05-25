import json
import numpy as np

with open('categories_info.json') as categories_file:
    categories = json.load(categories_file)
    
with open('labels.json') as labels_file:
    labels = json.load(labels_file)
    
with open('images_info.json') as images_info_file:
    images_info = json.load(images_info_file)

# image ID --> file_name    
image_files = {image['id']: image['file_name'] for image in images_info}

# category ID --> category_name
categories = {category['id']: category['name'] for category in categories}
category_ids = list(categories.keys())

n_classes = len(categories)

# vector of ones and zeros for multi label classification
targets = {image_id: np.zeros((n_classes)) for image_id in image_files.keys()}

for label in labels:
    targets[label['image_id']][category_ids.index(label['category_id'])] = 1.
    
""" Image Preprocessing """

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split

image_folder = './val2017/'

# image = load_img('./val2017/000000000139.jpg', target_size=(299, 299))

images = np.array([img_to_array(load_img(image_folder + image_file, target_size=(299, 299))) for image_file in image_files.values()])
targets = np.array([targets[image_id] for image_id in image_files.keys()])

x_train, x_validation, y_train, y_validation = train_test_split(images, targets, test_size=0.2, random_state=100)

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   rotation_range = 30, 
                                   zoom_range = 0.3, 
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2, 
                                   horizontal_flip = True,
                                   fill_mode = "nearest")
train_generator = train_datagen.flow(x_train, y_train, shuffle=True, batch_size=32, seed=10)

val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=32, seed=10)

""" Classifier Model """

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# fully-connected layer
x = Dense(1024, activation='relu')(x)

# output layer
predictions = Dense(n_classes, activation='sigmoid')(x)

# classifier model
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(Adam(lr=.0001), loss='binary_crossentropy', metrics=["accuracy"])

model.fit_generator(train_generator,
                      steps_per_epoch = len(x_train) // 32,
                      validation_data = val_generator,
                      epochs = 10,
                      verbose = 1)

model.save('classifier_model.h5')

# fine-tuning convolutional layers from inception V3

# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# freeze the first 249 layers and unfreeze the rest
#for layer in model.layers[:249]:
#   layer.trainable = False
#for layer in model.layers[249:]:
#   layer.trainable = True
#
#model.compile(Adam(lr=.0001), loss='binary_crossentropy', metrics=["accuracy"])
#
#model.fit_generator(train_generator,
#                      steps_per_epoch = len(x_train) // 32,
#                      validation_data = val_generator,
#                      epochs = 10,
#                      verbose = 2)
#
#model.save('clf_model_final.h5')

""" Test """

from keras.metrics import binary_accuracy
from keras.models import load_model

clf = load_model('classifier_model.h5')

test_image_folder = None

# image = load_img('./val2017/000000000139.jpg', target_size=(299, 299))

test_images = np.array([img_to_array(load_img(image_folder + image_file, target_size=(299, 299))) for image_file in image_files.values()])
test_targets = np.array([targets[image_id] for image_id in image_files.keys()])

test_images = test_images / 255.

test_pred = clf.predict(test_images)

accuracy = binary_accuracy(test_targets, test_pred)

print("Test accuracy = {}".format(accuracy))

#######################

image_path = None

image = np.array([img_to_array(load_img(image_path), target_size=(299, 299))])
image = image / 255.

pred = model.predict(image)

binary_pred = [p > 0.5 for p in pred]

labels = np.array(category_ids.values())[binary_pred]

print("Labels = {}".format(labels))