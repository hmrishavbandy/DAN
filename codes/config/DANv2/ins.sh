apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1

pip install tensorflow-gpu==2.5.0
pip install onnx-tf==1.8.0
pip install tensorflow_addons==0.13.0
