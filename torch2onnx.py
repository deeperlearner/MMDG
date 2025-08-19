import os
import argparse

import numpy as np
import onnx
import torch

from models.ViT import ViT_AvgPool_2modal_CrossAtten_Channel


SIZE = 256
def convert_onnx(net, path_module, output, opset=14, simplify=False):
    assert isinstance(net, torch.nn.Module)
    R_img = np.random.randint(0, 255, size=(SIZE, SIZE, 3), dtype=np.int32)
    R_img = R_img.astype(float)
    R_img = (R_img / 255. - 0.5) / 0.5  # torch style norm
    R_img = R_img.transpose((2, 0, 1))
    R_img = torch.from_numpy(R_img).unsqueeze(0).float()

    I_img = np.random.randint(0, 255, size=(SIZE, SIZE, 3), dtype=np.int32)
    I_img = I_img.astype(float)
    I_img = (I_img / 255. - 0.5) / 0.5  # torch style norm
    I_img = I_img.transpose((2, 0, 1))
    I_img = torch.from_numpy(I_img).unsqueeze(0).float()

    loaded_dict = torch.load(path_module, weights_only=True)
    net.load_state_dict(loaded_dict, strict=True)
    net.eval()
    img = (R_img, I_img)
    torch.onnx.export(net, img, output, input_names=['R_data', 'I_data'], output_names = ['output'],
                      dynamic_axes={'R_data': {0: 'batch_size'}, 'I_data': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    #graph = model.graph
    #graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('--input', type=str, help='input backbone.tar file or path')
    parser.add_argument('--output', type=str, help='output onnx path')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    print(args)

    Fas_Net = ViT_AvgPool_2modal_CrossAtten_Channel()
    convert_onnx(Fas_Net, args.input, args.output, simplify=args.simplify)
