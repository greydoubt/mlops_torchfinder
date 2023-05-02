import os
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, dataset_dir, img_height, img_width, batch_size):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def load_train_data(self):
        train_data_dir = os.path.join(self.dataset_dir, 'train')
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            train_data_dir,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset='training'
        )
        return train_data

    def load_validation_data(self):
        validation_data_dir = os.path.join(self.dataset_dir, 'train')
        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            validation_data_dir,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset='validation'
        )
        return validation_data

    def load_test_data(self):
        test_data_dir = os.path.join(self.dataset_dir, 'test')
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_dir,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=False
        )
        return test_data
