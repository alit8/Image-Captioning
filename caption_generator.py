import json
import string

""" Caption Preprocessing """

with open('captions.json') as captions_file:
    captions = json.load(captions_file)
    
with open('images_info.json') as images_info_file:
    images_info = json.load(images_info_file)

with open('categories_info.json') as categories_file:
    categories = json.load(categories_file)
    
with open('labels.json') as labels_file:
    labels = json.load(labels_file)

# image ID --> file_name    
image_files = {image['id']: image['file_name'] for image in images_info}
    
image_ids = list(image_files.keys()) # list of image IDs

# image ID --> list of captions
descriptions = {image_id: [] for image_id in image_ids}

for caption in captions:
    descriptions[caption['image_id']].append(caption['caption'])

table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  'startseq ' + ' '.join(desc) + ' endseq'
        
# vocabulary = set()
# for key in descriptions.keys():
#     [vocabulary.update(d.split()) for d in descriptions[key]]
    
all_captions = []
for key, val in descriptions.items():
    for cap in val:
        all_captions.append(cap)

# Consider only words which occur at least 10 times in all captions
word_count_threshold = 10
word_counts = {}

for sent in all_captions:
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

# Vocabulary of frequent words
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
vocab_size = len(vocab) + 1

""" Image preprocessing """

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
import numpy as np

image_folder = './val2017/'

# image = load_img('./val2017/000000000139.jpg', target_size=(299, 299))

images = np.array([img_to_array(load_img(image_folder + image_file, target_size=(299, 299))) for image_file in image_files.values()])

images_ = images / 255.

""" Extracting feature vectors for images """

import pickle
from keras.models import load_model
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load classifier model
model = load_model('classifier_model.h5')

# create a new model without considering output layer of classifier model
model_new = Model(model.input, model.layers[-2].output)

# extract feature vectors
feature_vectors = model_new.predict(images_)

# image IDs --> feature vector
temp_zip = zip(image_ids, feature_vectors)
features_dict = dict(temp_zip)

# save feature vectors
pickle_out = open("feature_vectors.pickle", "wb")
pickle.dump(features_dict, pickle_out)
pickle_out.close()

#pickle_in = open("dict.pickle","rb")
#example_dict = pickle.load(pickle_in)

# index --> word
ixtoword = {}
# word --> index
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
    
# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    
    all_desc = list()
    
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    
    return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(descriptions)
print('Max Description Length: %d' % max_length)

def data_generator(descriptions, features, wordtoix, max_length):
    X1, X2, y = list(), list(), list()
    
    for key, desc_list in descriptions.items():
        # retrieve the image feature
        image = features[key]
        for desc in desc_list:
            # encode the sequence
            seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(image)
                X2.append(in_seq)
                y.append(out_seq)
                
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = data_generator(descriptions, features_dict, wordtoix, max_length)
X1_train, X1_validation, X2_train, X2_validation, y_train, y_validation = train_test_split(X1, X2, y, test_size=0.2, random_state=100)

# Load Glove vectors
embeddings_index = {}
f = open('glove.6B.200d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
# Get 200-dim dense vector for each of words in vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

""" Final Model """
        
from keras.engine.input_layer import Input
from keras.layers import Dropout, Dense, Embedding, LSTM, Add
        
# image feature extractor model
inputs1 = Input(shape=(1024,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# partial caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# decoder (feed forward) model
decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x=[X1_train, X2_train], y=y_train, batch_size=64, epochs=5, verbose=1, validation_data=([X1_validation, X2_validation], y_validation))

model.save("image_captioner_model.h5")

""" Test """

def greedySearch(image):
    in_text = 'startseq'
    
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = np.array(pad_sequences([sequence], maxlen=max_length))
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
        
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

image_path = './1.jpg'

image = img_to_array(load_img(image_path, target_size=(299, 299)))
image = image / 255.
image = np.array([image])

feature_vector = model_new.predict(image)

# image = np.array([features_dict[581781]])
caption = greedySearch(feature_vector)

print("Caption = {}".format(caption))