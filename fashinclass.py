import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

df = keras.datasets.fashion_mnist.load_data()

(X_train_f,y_train_f),(X_test,ytest) = df

# fig , axes = plt.subplots(nrows=3,ncols=3,figsize=(5,6))
# a = 0
#
# for i in range(3):
#     for j in range(3):
#         axes[i,j].imshow(X_train_f[a],cmap=plt.get_cmap('gray'))
#
#         a = a +1
# plt.show()

X_train, y_train = X_train_f[5000:] / 255.0, y_train_f[5000:]
X_valid, y_valid = X_train_f[:5000] / 255.0, y_train_f[:5000]
# X_test = X_test / 255


# print(X_valid[0])

class_name = ['t-shirt/top','trousers','puliver','drees','coat','sandal','shirt','sneaker','bag','boot']

# print(class_name[5])


model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300,activation='relu'),
    keras.layers.Dropout(rate=0.01),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dropout(rate=0.01),
    keras.layers.Dense(10,activation='softmax')

])
# print(model.summary())
# hi1 = model.layers[1]
# w = hi1.get_weights()

model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs= 32 ,validation_data=(X_valid,y_valid),batch_size=32)

pd.DataFrame(history.history).plot(figsize=(9,6))
plt.grid(True)
plt.show()

print(model.evaluate(X_test,ytest))
X_new = X_test[:3]
y_prid= model.predict(X_new)
y_class = y_prid.argmax(axis=-1)
print(y_class)