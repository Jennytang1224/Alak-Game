import os
import pickle

import tensorflow as tf
import numpy as np

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# -------------> The following three are new imports <------------------
import time

np.set_printoptions(formatter={'float': '{:.5f}'.format})


class Model:

    def load_data(self, data_fname, label_fname):
        with open(data_fname, 'rb') as handle:
            data = pickle.load(handle)
            print(data.shape)
        with open(label_fname, 'rb') as handle:
            label = pickle.load(handle)
            print(label.shape)
        print("load data to file is completed")
        return data, label

    def create_tfmodel(self, X_train, y_train, X_val, y_val, checkpoint_path, checkpoint_dir):
        # tf.random.set_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dropout(0.2), # randomly erase 20% of the values -> prevent overfitting
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),  # randomly erase 20% of the values -> prevent overfitting
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=.1),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        )

        # Create a callback object
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history = model.fit(X_train, y_train, epochs=11,
                  validation_data=(X_val, y_val), callbacks=[cp_callback])
        os.listdir(checkpoint_dir)
        return model, history

    def create_skmodel(self, X_train, y_train, X_val, y_val):

        # Train a NN classification model
        print("Fitting the classifier to the training set")
        t0 = time.time()
        mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256, 100), activation='tanh',\
                            random_state=42, max_iter=100, learning_rate_init=0.1)

        parameters = {'learning_rate_init': [0.001, 0.01, 0.1, 0.5],\
                      'hidden_layer_sizes': [(100, 256, 100), (100, 500, 200), (100, 20)],\
                      'max_iter': [100, 1000]}

        clf = GridSearchCV(mlp_clf, parameters)
        clf.fit(X_train, y_train)

        print("Training done in {:0.3f}s".format(time.time() - t0))
        print("Training score: {:.4f}".format(clf.score(X_train, y_train)))
        t0 = time.time()
        # y_pred = clf.predict(X_val)
        print("Validation done in {:0.3f}s".format(time.time() - t0))
        print("Test score: {:.4f}".format(clf.score(X_val, y_val)))
        return clf

    def save_model(self, model, model_fname):
        if model_fname[-1] == '5': # tensorflow: .h5
            model.save(model_fname)
        else: # sklearn : pkl
            with open(model_fname, 'wb') as f:
                pickle.dump(model, f)

    def evaluate_model(self, model, type, X_test, y_test):
        if type == 'tf':
            print("Test score: {:.4f}".format(model.evaluate(X_test, y_test, verbose=2)))
        else:
            print("Test score: {:.4f}".format(model.score(X_test, y_test)))

    def run(self, type, model_fname, X_train, y_train, X_val, y_val, X_test, y_test):
        if type == 'sk':
            sk_model = model.create_skmodel(X_train, y_train, X_val, y_val)
            model.save_model(sk_model, model_fname)
            model.evaluate_model(sk_model, 'sk', X_test, y_test)

        else:
            checkpoint_path = "training/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            tf_model, history = model.create_tfmodel(X_train, y_train, X_val, y_val, checkpoint_path, checkpoint_dir)
            model.save_model(tf_model, model_fname)

            model.evaluate_model(tf_model, 'tf', X_train, y_train)
            model.evaluate_model(tf_model, 'tf', X_val, y_val)
            model.evaluate_model(tf_model, 'tf', X_test, y_test)


if __name__ == "__main__":
    model = Model()
    X_train, y_train = model.load_data('data/alak_data_may_11_v0.pickle', 'data/alak_label_may_11_v0.pickle')
    X_val, y_val = model.load_data('data/alak_data_may_11_v1.pickle', 'data/alak_label_may_11_v1.pickle')
    X_test, y_test = model.load_data('data/alak_data_may_11_v2.pickle', 'data/alak_label_may_11_v2.pickle')

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.run('sk', "models/alak_model_v4.pkl", X_train, y_train, X_val, y_val, X_test, y_test)
    # model.run('tf', "models/alak_model_v4.h5", X_train, y_train, X_val, y_val, X_test, y_test)

    # # Evaluate the model
    # # loss_untrain, acc_untrain = model.evaluate_model(tf_model, X_test, y_test)
    # # print("Using Untrained model, accuracy: {:5.2f}%".format(100 * acc_untrain))
    # #
    # # tf_model.load_weights(checkpoint_path) # Loads trained weights
    # # loss_train, acc_train = model.evaluate_model(tf_model, X_test, y_test)
    # # print("Using trained weights, accuracy: {:5.2f}%".format(100 * acc_train))






