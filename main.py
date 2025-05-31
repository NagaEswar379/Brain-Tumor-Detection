import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_gradcam(model, image, layer_name="conv2d_3"):  # Modify `layer_name` according to your model
    # Ensure the model is in inference mode
    model_inference = tf.keras.models.Model(inputs=model.input, outputs=model.output)

    # Get the last convolutional layer
    conv_layer = model.get_layer(layer_name)

    # Compute the gradient of the class prediction with respect to the feature map of the convolutional layer
    with tf.GradientTape() as tape:
        tape.watch(conv_layer.output)
        predictions = model_inference(image)
        class_idx = np.argmax(predictions[0])  # Get class index for max probability
        class_output = predictions[:, class_idx]  # Get the class prediction

    # Compute the gradient of the class output with respect to the convolutional layer
    grads = tape.gradient(class_output, conv_layer.output)

    # Compute the global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the feature map of the last convolutional layer
    conv_output = conv_layer.output[0]

    # Multiply each feature map by the corresponding pooled gradient
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = np.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU to keep only positive values
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]

    return heatmap

def overlay_heatmap_on_image(heatmap, original_image, alpha=0.4):
    # Resize heatmap to the same size as the original image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Convert the heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay the heatmap onto the original image
    superimposed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

def show_gradcam(model, image_path, layer_name="conv2d_3"):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))  # Resize according to model input size
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Compute Grad-CAM
    heatmap = compute_gradcam(model, img_array, layer_name)

    # Overlay the heatmap on the original image
    superimposed_img = overlay_heatmap_on_image(heatmap, img_rgb)

    # Display the image
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()
