from onnx_tf.backend import prepare
import onnx
onnx_model = onnx.load("./out_mod.onnx" )
tf_rep = prepare(onnx_model)  
tf_rep.export_graph("./out_tf.pb" )
