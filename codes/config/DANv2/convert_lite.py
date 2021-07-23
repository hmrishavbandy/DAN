import tensorflow as tf


converter=tf.compat.v1.lite.TFLiteConverter.from_saved_model("./out_tf.pb")
converter.allow_custom_ops=True

converter.experimental_new_converter = True
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open("out_lite.tflite", "wb").write(tflite_model)
