# From https://keras.io/examples/vision/visualizing_what_convnets_learn/
import numpy as np
import tensorflow as tf
import math
from tensorflow import keras
from IPython.display import Image, display

def compute_loss(input_image, filter_index, layer, model):
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

#@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, layer, model):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, layer, model)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image(img_width, img_height):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def get_visualize_filter(filter_index, img_width, img_height, layer, model):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image(img_width, img_height)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, layer, model)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

def visualize_filters(img_width, img_height, layer_name, model):
    # Set up a model that returns the activation values for our target layer
    layer = model.get_layer(name=layer_name)

    loss, img = get_visualize_filter(0, img_width, img_height, layer, model)
    keras.preprocessing.image.save_img("0.png", img)
    display(Image("0.png"))

    # Compute image inputs that maximize per-filter activations
    # for all filters of our target layer
    all_imgs = []
    number_of_filters = layer.kernel.shape[-1]
    if number_of_filters > 64:
        number_of_filters = 64
    for filter_index in range(number_of_filters):
        print("Processing filter %d" % (filter_index,))
        loss, img = get_visualize_filter(filter_index, img_width, img_height, layer, model)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our nxn filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = math.floor(math.sqrt(number_of_filters))
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_imgs[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height, :,] = img
    keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)
    display(Image("stiched_filters.png"))


