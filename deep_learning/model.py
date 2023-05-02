import tensorflow as tf


class Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Add model layers here
        outputs = self.PathFindingCNN(inputs)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, batch_size, epochs, validation_data):
        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
        )

    def predict(self, x):
        return self.model.predict(x)

    def PathFindingCNN(self, inputs):
        # Add CNN layers here
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(units=self.num_classes, activation="softmax")(x)
        
        return outputs
