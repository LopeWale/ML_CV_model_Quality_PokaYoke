"""
Create a custom convolutional neural network class that can be used to train an object detection model to using specify images and then use the weights from the first model in another model to detect many instances of object trained in the first model in other different images with a object segmentation model with high MAP accuracy and low computational cost and fast computational speed
"""

"""
Load all the libraries
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation
from keras.utils import plot_model
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import argparse
import sys
from google.colab import files
from google.colab import auth
from oauth2client.client import GoogleCredentials
import glob
import zipfile
import h5py
from skimage import exposure
from PIL import Image as pil_image
import matplotlib.pyplot as plt
import random
from shutil import copyfile
from sklearn.utils import shuffle
from tqdm import tqdm
"""
Create convolutional neural newtork class which can used to train and test the model
"""
class cnn_model(object):
    def __init__(self, input_shape, path, weights_path=None, weights_path_gen=None, batch_size=128, epochs = 100):
        """
        Initialise the variables and create the model architecture
        """
        self.input_shape = input_shape
        self.path = path
        self.weights_path = weights_path
        self.weights_path_gen = weights_path_gen
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_img = Input(shape=input_shape)
        self.conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1a")(self.input_img)
        self.conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1b")(self.conv1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(self.conv2)
        self.conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2a")(self.pool1)
        self.conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2b")(self.conv3)
        self.pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(self.conv4)
        self.conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',         name="conv3a")(self.pool2)
        self.conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3b")(self.conv5)
        self.pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(self.conv6)
        self.conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv4a")(self.pool3)
        self.conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv4b")(self.conv7)
        self.pool4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(self.conv8)
        self.conv9 = Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5a")(self.pool4)
        self.conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5b")(self.conv9)
        self.up1 = UpSampling2D((2, 2), name="up1")(self.conv10)
        self.conv11 = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv6a")(self.up1)
        self.conv12 = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv6b")(self.conv11)
        self.up2 = UpSampling2D((2, 2), name="up2")(self.conv12)
        self.conv13 = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv7a")(self.up2)
        self.conv14 = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv7b")(self.conv13)
        self.up3 = UpSampling2D((2, 2), name="up3")(self.conv14)
        self.conv15 = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv8a")(self.up3)
        self.conv16 = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv8b")(self.conv15)
        self.up4 = UpSampling2D((2, 2), name="up4")(self.conv16)
        self.conv17 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv9a")(self.up4)
        self.conv18 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv9b")(self.conv17)
        self.conv19 = Conv2D(2, (3, 3), activation='relu', padding='same', name="conv10")(self.conv18)
        self.conv22 = Conv2D(1, (1, 1), activation='sigmoid', name="conv11")(self.conv19)
        self.model = Model(self.input_img, self.conv22, name="detection_model")
        self.model_gen = Model(self.input_img, self.conv19, name="detection_model_gen")
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model_gen.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        """
        Print the model summary
        """
        self.model.summary()
        self.model_gen.summary()

    def fit_model(self, x, y, x_val=None, y_val=None):
        """
        Train the model
        """
        tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        save_model_path = self.path + 'detection_model'
        save_model_path_gen = self.path + 'detection_model_gen'
        checkpoint = ModelCheckpoint(save_model_path + '.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        checkpoint_gen = ModelCheckpoint(save_model_path_gen + '.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=[tbCallBack, checkpoint, early], validation_data=None)
        self.model_gen.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=[tbCallBack, checkpoint_gen, early], validation_data=None)


    def predict(self, X):
        """
        Predict the masks of images of unknown data
        """
        return self.model.predict(X)

    def predict_gen(self, X):
        """
        Predict the masks of images of unknown data
        """
        return self.model_gen.predict(X)

    def get_model(self):
        """
        Return the models
        """
        return [self.model, self.model_gen]

    def load_weights(self):
        """
        Load weights of the model
        """
        self.model = load_model(self.weights_path)
        self.model_gen = load_model(self.weights_path_gen)

    def export_model(self, path, image_shape, weights_path, model_name="detection_model"):
        """
        Export the model to tensorflow
        """
        self.model = load_model(weights_path)
        tf.train.write_graph(self.model.graph, path, model_name + ".pb", as_text=False)
        freeze_graph.freeze_graph(os.path.join(path, model_name + '.pb'), "", False, os.path.join(path, weights_path), model_name + "/conv11/Sigmoid", "save/restore_all", "save/Const:0", os.path.join(path, model_name + '-frozen.pb'), True, "")
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(os.path.join(path, model_name + '-frozen.pb'), "rb") as f:
            input_graph_def.ParseFromString(f.read())
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, ["conv1a/Relu", "conv1b/Relu", "pool1"], [model_name + "/conv11/Sigmoid"], tf.float32.as_datatype_enum)
        with tf.gfile.FastGFile(os.path.join(path, model_name + '-frozen-optimized.pb'), "w") as f:
            f.write(output_graph_def.SerializeToString())
        input_node_names = ["conv1a/Relu", "conv1b/Relu", "pool1"]
        output_node_names = model_name + "/conv11/Sigmoid"
        tf.train.write_graph(output_graph_def, path, model_name + '.pbtxt', as_text=True)
        tf.train.write_graph(output_graph_def, path, model_name + '.pb', as_text=False)
        label_map = {0: "background", 1: "object"}
        with open(os.path.join(path, model_name + '.pbtxt'), "r") as f:
            pbtxt_content = f.read()
        for i in range(len(input_node_names)):
            pbtxt_content = pbtxt_content.replace("name: \"{}\"".format(input_node_names[i]),
                                                  "name: \"{}\"\n  shape {{\n    dim {{\n      size: {}\n    }}\n  }}".format(
                                                      input_node_names[i], image_shape[i]))
        with open(os.path.join(path, model_name + '.pbtxt'), "w") as f:
            f.write(pbtxt_content)

    def export_model_gen(self, path, image_shape, weights_path, model_name="detection_model_gen"):
        """
        Export the model to tensorflow
        """
        self.model_gen = load_model(weights_path)
        tf.train.write_graph(self.model_gen.graph, path, model_name + ".pb", as_text=False)
        freeze_graph.freeze_graph(os.path.join(path, model_name + '.pb'), "", False, os.path.join(path, weights_path), model_name + "/conv11/Sigmoid", "save/restore_all", "save/Const:0", os.path.join(path, model_name + '-frozen.pb'), True, "")
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(os.path.join(path, model_name + '-frozen.pb'), "rb") as f:
            input_graph_def.ParseFromString(f.read())
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, ["conv1a/Relu", "conv1b/Relu", "pool1"], [model_name + "/conv11/Sigmoid"], tf.float32.as_datatype_enum)
        with tf.gfile.FastGFile(os.path.join(path, model_name + '-frozen-optimized.pb'), "w") as f:
            f.write(output_graph_def.SerializeToString())
        input_node_names = ["conv1a/Relu", "conv1b/Relu", "pool1"]
        output_node_names = model_name + "/conv11/Sigmoid"
        tf.train.write_graph(output_graph_def, path, model_name + '.pbtxt', as_text=True)
        tf.train.write_graph(output_graph_def, path, model_name + '.pb', as_text=False)
        label_map = {0: "background", 1: "object"}
        with open(os.path.join(path, model_name + '.pbtxt'), "r") as f:
            pbtxt_content = f.read()
        for i in range(len(input_node_names)):
            pbtxt_content = pbtxt_content.replace("name: \"{}\"".format(input_node_names[i]),
                                                  "name: \"{}\"\n  shape {{\n    dim {{\n      size: {}\n    }}\n  }}".format(
                                                      input_node_names[i], image_shape[i]))
        with open(os.path.join(path, model_name + '.pbtxt'), "w") as f:
            f.write(pbtxt_content)


    def accuracy(self, y_pred, y_label):
        """
        Returns the percentage of correct predictions
        """
        y_pred = y_pred > 0.5
        num_correct = 0
        for i in range(len(y_pred)):
            if np.array_equal(y_pred[i], y_label[i]):
                num_correct += 1
        prcnt_corr = num_correct / len(y_pred)
        return prcnt_corr
    """
    METHODS TO READ THE DATA AND MODIFY IT
    """
    def load_data(self, path, image_format='*.JPEG', split=False, set_extension="", max_count=100, stratify=None, classes_to_consider=None):
        """
        Loads the data
        """
        if classes_to_consider == None:
            img_name_collection = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], image_format))]
            img_count = len(img_name_collection)
            img_name_collection = shuffle(img_name_collection)
            img_name_collection = img_name_collection[:max_count]
            images = []
            for img in img_name_collection:
                images.append(img_to_array(load_img(img)))
            X = np.array(images, dtype="float")
            if X.shape[0] == 1:
                X = np.expand_dims(X, axis=0)
            else:
                X /= 255
            if split == True:
                X_train = X[:int(len(X)*0.8)]
                X_test = X[int(len(X)*0.8):]
                return [X_train, X_test]
            return X
        else:
            classes = os.listdir(path)
            img_name_collection = []
            for cls in classes:
                img_name_collection.append([y for x in os.walk(path+cls) for y in glob.glob(os.path.join(x[0], image_format))])
            if classes_to_consider != None:
                for i in range(len(img_name_collection)):
                    if classes_to_consider[i] == 0:
                        img_name_collection[i] = [os.path.join(img_name_collection[i][j], "not_class") for j in range(len(img_name_collection[i]))]
            all_img_name_collection = []
            for i in range(len(img_name_collection)):
                for j in range(len(img_name_collection[i])):
                    all_img_name_collection.append(img_name_collection[i][j])
            for i in range(len(img_name_collection)):
                for j in range(len(img_name_collection[i])):
                    if img_name_collection[i][j].split("/")[-2] != "not_class":
                        img = load_img(img_name_collection[i][j])
                        width = img.size[0]
                        height = img.size[1]
                        pix = img.load()
                        for x in range(width):
                            for y in range(height):
                                if pix[x, y] == 0:
                                    pix[x,y] = 255
                        img.save(img_name_collection[i][j])
            all_img_name_collection = shuffle(all_img_name_collection)
            img_count = len(all_img_name_collection)
            img_name_collection = [all_img_name_collection[i] for i in range(img_count)]
            images = []
            for img in img_name_collection:
                images.append(img_to_array(load_img(img)))
            Y = np.array([0 if img.split("/")[-2] == "not_class" else 1 for img in all_img_name_collection])
            X = np.array(images, dtype="float")
            if X.shape[0] == 1:
                X = np.expand_dims(X, axis=0)
            else:
                X /= 255
            if split == True:
                X_train = X[:int(len(X)*0.8)]
                X_test = X[int(len(X)*0.8):]
                y_train = Y[:int(len(X)*0.8)]
                y_test = Y[int(len(X)*0.8):]
                return [X_train, X_test, y_train, y_test]
            return [X, Y]

    def generate_data(self, X, y, batch_size, augment=False, augment_count=3, augment_magnitude=0.1):
        """
        Generator the data with the help of keras image datagenerator module
        """
        if augment == False:
            datagen = ImageDataGenerator()
            while 1:
                for i in range(0, len(X), batch_size):
                    yield datagen.flow(X[i:i+batch_size], y[i:i+batch_size], shuffle=False, batch_size=batch_size)
        else:
            datagen = ImageDataGenerator(height_shift_range=augment_magnitude, width_shift_range=augment_magnitude, rotation_range=augment_magnitude*200, horizontal_flip=True, vertical_flip=True)
            while 1:
                for i in range(0, len(X), batch_size):
                    augmented_data = []
                    augmented_labels = []
                    X_batch = []
                    y_batch = []
                    for num in range(batch_size*augment_count):
                        if num % augment_count == 0:
                            X_batch.append(X[(i+num)%len(X)])
                            y_batch.append(y[(i+num)%len(X)])
                    X_batch = np.array(X_batch)
                    X_batch = np.tile(X_batch, (augment_count, 1, 1, 1))
                    y_batch = np.array(y_batch)
                    y_batch = np.repeat(y_batch, augment_count, axis=0)
                    for j in range(augment_count):
                        for batch in datagen.flow(X_batch[j:j+batch_size], y_batch[j:j+batch_size], shuffle=False, batch_size=batch_size):
                            augmented_data.append(batch[0])
                            augmented_labels.append(batch[1])
                            break
                    augmented_data = np.array(augmented_data)
                    augmented_labels = np.array(augmented_labels)
                    yield [augmented_data, augmented_labels]

    def split_data(self, X, y, batch_size, val_split=0.1):
        """
        Split the data into training and test set
        """
        X_train = X[:int(len(X)*(1-val_split))]
        X_test = X[int(len(X)*(1-val_split)):]
        y_train = y[:int(len(X)*(1-val_split))]
        y_test = y[int(len(X)*(1-val_split)):]
        return [X_train, X_test, y_train, y_test]

    def pre_process_labels(self, y, augmentation=True):
        """
        Create valid masks for the training data which have the size of the input image and have 1 as the object and 0 as the background
        """
        processed_labels = []
        for label in y:
            processed_label = np.zeros(label.size)
            processed_label[np.where(label == 1)[0]] = 1
            processed_labels.append(processed_label)
        return np.array(y)

    def pre_process_images(self, X, augmentation=True):
        """
        Preprocess the images
        """
        processed_images = []
        for i in range(len(X)):
            if augmentation==True:
                if random.random() > 0.5:
                    processed_images.append(rotation(X[i]))
                else:
                    if random.random() > 0.5:
                        processed_images.append(flip_vertically(X[i]))
                    else:
                        processed_images.append(flip_horizontally(X[i]))
            else:
                processed_images.append(X[i])

        return np.array(processed_images)
    """
    METHODS TO PREPROCESS THE LABELS
    """
    def flip_vertically(self, X):
        """
        Flip image vertically
        """
        return np.fliplr(X)

    def flip_horizontally(self, X):
        """
        Flip image horizontally
        """
        return np.flipud(X)

    def rotation(self, X):
        """
        Rotate images
        """
        return np.rot90(X)
"""
Mount the drive
"""
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/open?id=1dkz_KOjlxE21sTSnPAtYaV7dmLnuKX9V'
_, id = link.split("=")
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('train-jpg.tar.7z')
files.download('train-jpg.tar.7z')
!7z x -aoa -o../content/ ../content/train-jpg.tar.7z
"""
Created the directories if they don't exist
"""
path = "../content/"
if not os.path.exists(path):
    os.makedirs(path)
path_train = "../content/train/"
if not os.path.exists(path_train):
    os.makedirs(path_train)
"""
Copy the data from zip file to created file
"""
file = zipfile.ZipFile(path + "train-jpg" + ".tar.7z")
file.extractall(path + "train-jpg")
"""
Check if the data copied correctly
"""
!ls ../content/train-jpg

"""
Create train and test sets
"""
path = "../content/train-jpg/"
input_shape = (128, 128, 3)
model = cnn_model(input_shape, path)
X = model.load_data(path, image_format='*.jpg')
X = model.pre_process_images(X)
y = model.load_data(path, set_extension="/masks/", image_format='*.png')
y = model.pre_process_labels(y)
[X_train, X_test, y_train, y_test] = model.split_data(X, y, 64)
"""
Create the model and train it
"""
input_shape = (128, 128, 3)
model = cnn_model(input_shape, path)
model.fit_model(X_train, y_train)
"""
Test the model
"""
model.load_weights()
y_pred = model.predict(X_test)
prcnt_corr = model.accuracy(y_pred, y_test)
print("Accuracy: %", prcnt_corr)
"""
Visualise the model
"""
K.clear_session()
plot_model(model.get_model(), to_file=path + "cnn_mask.png", show_shapes=True)
files.download(path+"cnn_mask.png")
"""
Export the tensorflow model
image_shape = (None, None, 3)
model.export_model(path, image_shape, weights_path, "detection_model")
model.export_model(path, image_shape, weights_path, "detection_model_gen")
"""
"""
Download the model files
"""
files.download(path + "detection_model.pb")
files.download(path + "detection_model.pbtxt")
files.download(path + "detection_model_gen.pb")
files.download(path + "detection_model_gen.pbtxt")
"""
Compress the model
"""
!tar -czvf ../content/detection_model.tgz ../content/detection_model.pb ../content/detection_model.pbtxt ../content/detection_model_gen.pb ../content/detection_model_gen.pbtxt
"""
Download the compressed file
"""
files.download(path + "detection_model.tgz")
"""
Clear all
"""
K.clear_session()
"""
Upload the file to gdrive
"""
f = drive.CreateFile()
f.SetContentFile(path + "google.pd.tgz")
f.Upload()
 

!tar -xvzf ../content/train-jpg.tar.7z

 
 
files.download('train-jpg.tar.7z')
input_shape = (128, 128, 3)
K.clear_session()
model = cnn_model(input_shape, path)
model.load_weights()
y_pred = model.predict(X)
prcnt_corr = model.accuracy(y_pred, y)
print("Accuracy: %", prcnt_corr)
 
 
model_path = path + "detection_model"
files.download(model_path + '.h5')
files.upload(model_path + '.h5')


!zip -r ../content/guava.zip ../content/guava/

!zip -r ../content/guava_segment.zip ../content/guava_segment/

!zip -r ../content/train-jpg.zip ../content/train-jpg/

!zip -r ../content/gu








