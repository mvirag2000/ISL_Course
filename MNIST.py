import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
from os import path, getcwd, chdir
from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

def build_model(n_hidden=1, n_neurons=256, learning_rate=0.001):
    print("\n*** Hidden: "+ str(n_hidden))
    print("*** Neurons: " + str(n_neurons))
    print("*** Learning Rate: " + str(learning_rate))
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = build_model(1, 100, 0.01) 
history = model.fit(x_train, y_train, epochs=100, verbose=3, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)])

parms = model.count_params()
loss = history.history['loss'][-1]

y_pred = model.predict(x_train).argmax(axis=1)
train_acc = accuracy_score(y_train, y_pred)

y_pred = model.predict(x_test.reshape((-1,28,28))).argmax(axis=1)
test_acc = accuracy_score(y_test, y_pred)

print("\nRESULTS")
print("Train accuracy: " + str(train_acc))
print("Test accuracy: " + str(test_acc))
print("Parameters: " + str(parms))
print("Training loss: " + str(loss))

#plt.plot(history.history['loss'], label='Train')
## plt.plot(history.history['val_loss'], label='Test')
#plt.title('Model Loss')
# plt.legend(loc='lower left')
#plt.ylabel('Entropy')
#plt.xlabel('epoch')
#plt.show()