##
## I am going to try the double descent thing using MNIST and a simple NN
##
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'} 
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
import seaborn as sns
sns.set_theme() 

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

def build_model(n_hidden=1, n_neurons=256, learning_rate=0.001):
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

test_acc_list = []
parms_list = []
test_loss_list = []
train_loss_list = []

for nodes in np.arange(10, 61, 2):

    model = build_model(1, nodes, 0.01) 
    history = model.fit(x_train, y_train, epochs=100, verbose=3, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)])
    parms = model.count_params()
 
    y_pred = model.predict(x_train)
    train_loss = log_loss(y_train, y_pred) 
    train_acc = accuracy_score(y_train, y_pred.argmax(axis=1))

    y_pred = model.predict(x_test)
    test_loss = log_loss(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred.argmax(axis=1))

    print("\nRESULTS")
    print("Train accuracy: " + str(train_acc))
    print("Test accuracy: " + str(test_acc))
    print("Parameters: " + str(parms))
    print("Training loss:" + str(train_loss)) 
    
    test_acc_list.append(test_acc)
    parms_list.append(parms)
    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)  

print(train_loss_list)
print(test_loss_list)
print(parms_list)
print(test_acc_list)


fig, ax1 = plt.subplots()
ax1.set_xlabel('Parameters')
ax1.set_ylabel('Log Loss')
ax1.plot(parms_list, train_loss_list, label='Train', color='green')
ax1.plot(parms_list, test_loss_list, label='Test', color='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Test Accuracy')
ax2.plot(parms_list, test_acc_list, label='Accuracy', color='red')

ax1.legend()
fig.tight_layout()
plt.show()