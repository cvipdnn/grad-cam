import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.python.framework import ops


classifier_layer_names = [
    "block5_pool",
    "flatten",
    "fc1",
    "fc2",
    "predictions",
]

# Define model here ---------------------------------------------------
def build_model():
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    return VGG16(include_top=True, weights='imagenet')

H, W = 224, 224 # Input shape, defined by the model (model.input_shape)
# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy 
  return tf.nn.relu(x), grad

def build_guided_model(model, layer_name):
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """

    gb_model = tf.keras.Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(layer_name).output]
    )

    layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu

    return gb_model


def guided_backprop(gb_model, images):
    """Guided Backpropagation method for visualizing input saliency."""
    

    with tf.GradientTape() as tape:
        inputs = tf.cast(images, tf.float32)
        tape.watch(inputs)
        outputs = gb_model(inputs)
    

    gb = tape.gradient(outputs,inputs)
    gb = np.array(gb)


    return gb


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""

    last_conv_layer = input_model.get_layer(layer_name)
    last_conv_layer_model = tf.keras.Model(input_model.inputs, last_conv_layer.output)


    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x=classifier_input
    
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    classifier_model = tf.keras.Model(classifier_input, x)


    with tf.GradientTape() as tape:
        images_tensor = tf.cast(image, tf.float32)
        last_conv_layer_output = last_conv_layer_model(images_tensor)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        y_c = preds[:, cls]
        tape.watch(last_conv_layer_output)

    grads_val = tape.gradient(y_c, last_conv_layer_output)

    conv_output = last_conv_layer_output

    # grads = normalize(grads)
    
    output = last_conv_layer_output
 
    
    output, grads_val = output[0, :], grads_val[0, :, :, :]


    weights = np.mean(grads_val, axis=(0, 1))
    
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    return cam
    
def compute_saliency(model, gb_model, img_path, layer_name='block5_conv3', cls=-1, visualize=True, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
    top_n = 5
    top = decode_predictions(predictions, top=top_n)[0]
    classes = np.argsort(predictions[0])[-top_n:][::-1]
    print('Model prediction:')
    for c, p in zip(classes, top):
        print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    if cls == -1:
        cls = np.argmax(predictions)
    class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    print("Explanation for '{}'".format(class_name))
    
    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    
    gb = guided_backprop(gb_model, preprocessed_input)
    
    guided_gradcam = gb*gradcam[..., np.newaxis]

    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))
    
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))
        
        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()
        
    return gradcam, gb, guided_gradcam

if __name__ == '__main__':
    model = build_model()
    gbmodel = build_guided_model(model, layer_name='block5_conv3')
    gradcam, gb, guided_gradcam = compute_saliency(model, gbmodel, layer_name='block5_conv3',
                                             img_path=sys.argv[1], cls=-1, visualize=True, save=True)
