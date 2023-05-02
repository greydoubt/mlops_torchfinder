import tensorflow as tf


class Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Add model layers here

        outputs = tf.keras.layers.Dense(units=self.num_classes, activation="softmax")(x)

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


