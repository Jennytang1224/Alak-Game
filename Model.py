import os
import pickle

import tensorflow as tf


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

    def create_model(self, X_train, y_train, X_val, y_val, checkpoint_path, checkpoint_dir):
        tf.random.set_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2), # randomly erase 20% of the values -> prevent overfitting
            tf.keras.layers.Dense(256, activation='relu'),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        )

        # Create a callback object
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history = model.fit(X_train, y_train, epochs=15,
                  validation_data=(X_val, y_val), callbacks=[cp_callback])
        os.listdir(checkpoint_dir)
        return model, history

    def save_model(self, model, model_fname):
        model.save(model_fname)


    def evaluate_model(self, model, X_test, y_test):
        loss, acc = model.evaluate(X_test, y_test, verbose=2)
        return loss, acc


if __name__ == "__main__":
    model = Model()
    X_train, y_train = model.load_data('data/alak_data_may_6_v0.pickle', 'data/alak_label_may_6_v0.pickle')
    X_val, y_val = model.load_data('data/alak_data_may_6_v2.pickle', 'data/alak_label_may_6_v2.pickle')
    X_test, y_test = model.load_data('data/alak_data_may_6_v1.pickle', 'data/alak_label_may_6_v1.pickle')

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # create model
    tf_model, history = model.create_model(X_train, y_train, X_val, y_val, checkpoint_path, checkpoint_dir)
    model.save_model(tf_model, "models/alak_model_v1.h5")

    # Evaluate the model
    # loss_untrain, acc_untrain = model.evaluate_model(tf_model, X_test, y_test)
    # print("Using Untrained model, accuracy: {:5.2f}%".format(100 * acc_untrain))
    #
    # tf_model.load_weights(checkpoint_path) # Loads trained weights
    # loss_train, acc_train = model.evaluate_model(tf_model, X_test, y_test)
    # print("Using trained weights, accuracy: {:5.2f}%".format(100 * acc_train))

    loss_train, acc_train = model.evaluate_model(tf_model, X_train, y_train)
    print("Using trained model, training  accuracy: {:5.2f}%".format(100 * acc_train))

    loss_val, acc_val = model.evaluate_model(tf_model, X_val, y_val)
    print("Using trained model, validation accuracy: {:5.2f}%".format(100 * acc_val))

    loss_test, acc_test = model.evaluate_model(tf_model, X_test, y_test)
    print("Using trained model, test accuracy: {:5.2f}%".format(100 * acc_test))






