from pandas import read_csv
from PIL import Image
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import os 
import glob
from efficientnet.keras import center_crop_and_resize, preprocess_input


class rt():
    def __init__(self):
        self.lemon_place = os.path.dirname(__file__).rstrip('/src')
        self.train_images_place = self.lemon_place + "/datas/transform_lemons/train_images"
        DIR = self.train_images_place
        self.file_sum = sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR))


    def get_labels(self):
        place = self.lemon_place + "/datas/train_images.csv"
        df = read_csv(place,engine='python')
        df = df["class_num"]
        labels = df.values
        return labels

    def get_imgs(self):
        place=self.train_images_place
        num_file=self.file_sum
        img_list = []
        for i in range(num_file):
            file_name = '/' + str(i).zfill(4)
            img = np.array(cv2.imread(place + file_name + '.jpg'))
            image_size = 600 
            img = center_crop_and_resize(img, image_size=image_size)
            img = preprocess_input(img)
            #img = np.expand_dims(img, 0)
            img_list.append(img)
        return np.array(img_list)

    def run_train(self):
        img_list = self.get_imgs()
        label_list = self.get_labels()
        label_list = keras.utils.to_categorical(label_list)
        lemon_place = self.lemon_place
        #train_img,train_label,test_img,test_label = train_test_split(img_list,label_list)

        base_model = tf.keras.applications.EfficientNetB7(input_shape=(600,600,3), weights='imagenet', include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(4, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output]) 
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(img_list,label_list,epochs=20,validation_split=0.2)
        model.save(lemon_place+'/fine_model.h5')


if __name__ == "__main__":
    a = rt().get_imgs()
    b = rt().get_labels()
    print(np.shape(a))
    print(np.shape(b))
    #model.fit(X_train_processed, y_train_categorical)
