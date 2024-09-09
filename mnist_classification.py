import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values (0 to 255) to (0 to 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the input images from 28x28 to 1D vector
    layers.Dense(128, activation='relu'),  # First hidden layer with 128 neurons
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes (0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')

# Visualize some test images with predictions
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = tf.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f'{predicted_label} ({100 * tf.reduce_max(predictions_array):2.0f}%)', color=color)

# Plot first 5 test images
for i in range(5):
    plt.figure(figsize=(6,3))
    plot_image(i, predictions, test_labels, test_images)
    plt.show()
