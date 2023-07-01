import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras import metrics, optimizers
from keras.utils import to_categorical

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform target to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define metrics
metric_list = [metrics.MeanSquaredError(), metrics.BinaryAccuracy(), metrics.CategoricalAccuracy(), 
               metrics.AUC(), metrics.Precision(), metrics.Recall(), metrics.FalsePositives(), 
               metrics.FalseNegatives(), metrics.TruePositives(), metrics.TrueNegatives()]

# Optimizers
optimizers_dict = {
    'SGD': optimizers.SGD(),
    'RMSprop': optimizers.RMSprop(),
    'Adam': optimizers.Adam(),
    'Adadelta': optimizers.Adadelta(),
    'Adagrad': optimizers.Adagrad(),
    'Adamax': optimizers.Adamax(),
    'Nadam': optimizers.Nadam(),
    'Ftrl': optimizers.Ftrl()
}

# Loop through each optimizer
for opt_name, optimizer in optimizers_dict.items():
    print(f"Running with optimizer: {opt_name}")

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model with the current optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metric_list)

    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=10, verbose=0)

    # Plot the metrics for the current optimizer
    fig, axs = plt.subplots(len(metric_list), figsize=(10, 20))
    fig.suptitle(f'Metrics for {opt_name}')
    
    for ax, metric in zip(axs, metric_list):
        ax.plot(history.history[metric.name], label='train')
        ax.plot(history.history['val_' + metric.name], label='test')
        ax.set_title(metric.name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.name)
        ax.legend()
    
    # Save the plot to a file
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{opt_name}_metrics.png")

    plt.show()
