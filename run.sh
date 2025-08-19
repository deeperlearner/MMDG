#!/bin/bash

# python3 train.py --num_epoch 15
# python3 test.py

# torch to onnx
python3 torch2onnx.py --input ./saved/FASNet-epoch10.tar --output ./saved/FASNet-epoch10.onnx
