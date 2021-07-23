import numpy as np
import tensorflow as tf
import cv2
def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  image=tf.transpose(image, perm=[0, 3, 1, 2])

  return image

content_image = load_img("/root/hmrishav/DAN/im_cat.jpg")

pre_img = preprocess_image(content_image, 256)
print(pre_img.shape)
im=tf.transpose(pre_img, perm=[0, 2, 3, 1])
# cv2.imwrite("this.jpg",im.numpy()[0]*255)



# Load the TFLite model and allocate tensors.

interpreter = tf.lite.Interpreter(model_path="out_lite.tflite")
# interpreter.resize_tensor_input(0, [1, 256, 256, 3], strict=True)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()



interpreter.set_tensor(input_details[0]["index"], pre_img)
# interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
interpreter.invoke()


out_img = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()


print(out_img.shape)

