import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras

(train_img, train_label), (test_img, test_label) = keras.datasets.mnist.load_data()

train_img = train_img.reshape([-1, 28, 28, 1])
test_img = test_img.reshape([-1, 28, 28, 1])
train_img = train_img / 255.0
test_img = test_img / 255.0

train_label = keras.utils.to_categorical(train_label)
test_label = keras.utils.to_categorical(test_label)

model = keras.Sequential([
    keras.layers.Conv2D(32, 5,
                        strides=(1, 1),
                        activation='relu',
                        kernel_initializer='VarianceScaling',
                        input_shape=[28, 28, 1]),

    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    keras.layers.Conv2D(64, 3,
                        strides=(1, 1),
                        padding="same",
                        activation='relu',
                        kernel_initializer='VarianceScaling'),

    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu',
                       kernel_initializer='VarianceScaling'),

    keras.layers.Dense(10, activation='softmax',
                       kernel_initializer='VarianceScaling')
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.15),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_img,
          train_label,
          validation_data=(test_img, test_label),
          epochs=1)

test_loss, test_acc = model.evaluate(test_img, test_label)
print('Test accuracy:', test_acc)

tfjs.converters.save_keras_model(model, 'model')
