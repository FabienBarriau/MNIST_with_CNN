import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from MNIST_utils import load_MNIST
from MNIST_utils import load_MNIST_evaluation
from MNIST_utils import plot_confusion_matrix

CHECKPOINT_PATH = "training_1/cp.ckpt"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
RETRAIN = True

print("Begin")
train_images, train_labels, test_images, test_labels = load_MNIST()

train_images = train_images/255
test_images = test_images/255

#Plot images of MNIST:
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(train_labels[i])
# plt.show()

#Plot hist of labels to verify homogeinity
# plt.figure()
# plt.subplot(121)
# plt.hist(train_labels, density=True)
# plt.xlabel("train labels hist")
# plt.subplot(122)
# plt.hist(test_labels, density=True)
# plt.xlabel("test labels hist")
# plt.show()

#Need to expand dimension of the numpy to use convolutional layer.
train_images = np.expand_dims(train_images, axis=4)
test_images = np.expand_dims(test_images, axis=4)

#Convolutional Network based on leNet 5 with some improvement inspired by the AlexNet article: Use of relu instead
#of sigmoid as activation function, use of max pooling for subsampling layer and use of dropout to reduce complexity
#of the model.
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(filters=6, kernel_size=5, padding="same", input_shape=(28, 28, 1), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=16, kernel_size=5, padding="same", activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=120, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(units=84, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if(RETRAIN):
    model_train = create_model()
    model_train.summary()

    #Callback to avoid overfitting
    history_cnn = model_train.fit(train_images, train_labels, validation_split=0.2, epochs=20,
                                callbacks=[keras.callbacks.EarlyStopping(monitor="val_acc", min_delta=0.001, patience=4),
                                           tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True,
                                                                              verbose=1)])

    loss_retrain, acc_retrain = model_train.evaluate(test_images, test_labels)

    print('Test accuracy:', acc_retrain)

    #Loss and val_loss evolution
    plt.figure()
    plt.plot(history_cnn.history['loss'], 'b--')
    plt.plot(history_cnn.history['val_loss'], 'b-')
    plt.title("Loss Evolution")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()

    print(history_cnn.history['val_loss'])

model_evaluate = create_model()
model_evaluate.load_weights(CHECKPOINT_PATH)

#Verify if the accuracy is correct
loss_evaluate , acc_evaluate = model_evaluate.evaluate(test_images, test_labels)
test_proba_predicted = model_evaluate.predict(test_images)
test_labels_predicted = np.asarray([np.argmax(test_proba_predicted[i, :]) for i in range(test_proba_predicted.shape[0])])
print('Test accuracy:', acc_evaluate)

cm = confusion_matrix(test_labels, test_labels_predicted)
plot_confusion_matrix(cm, range(9), normalize=True)

images = load_MNIST_evaluation()
images = images/255

eval_proba_predicted = model_evaluate.predict(np.expand_dims(images, axis=4))
eval_labels_predicted = np.asarray([np.argmax(eval_proba_predicted[i, :]) for i in range(eval_proba_predicted.shape[0])])

#Plot images of MNIST evaluation with prediction
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(eval_labels_predicted[i])
plt.show()

df = pd.DataFrame({"ImageId": np.arange(1, eval_labels_predicted.shape[0]+1), "Label": eval_labels_predicted})
df.to_csv(SAMPLE_SUBMISSION_PATH, sep=",", index=False)
