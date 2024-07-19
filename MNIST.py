import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
from os import path, getcwd, chdir

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') is not None and logs.get('accuracy') >= 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
                
callbacks = myCallback()

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

def build_model(n_hidden=1, n_neurons=256, learning_rate=0.001):
    print("\n*** Hidden: "+ str(n_hidden))
    print("*** Neurons: " + str(n_neurons))
    print("*** Learning Rate: " + str(learning_rate))
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(28,28)))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1)) 
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    model.summary()
    return model

model = build_model(2, 100, 0.01) 
history = model.fit(x_train, y_train, epochs=5, verbose=2)
# history = model.fit(x_train, y_train, epochs=7, callbacks=[callbacks])

plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'], label='Test')
plt.title('Model Loss')
# plt.legend(loc='lower left')
plt.ylabel('Entropy')
plt.xlabel('epoch')
plt.show()