import onnxruntime as ort
import numpy as np
import cv2
import torch

from models.ViT import ViT_AvgPool_2modal_CrossAtten_Channel


def onnx_run(onnx_path, input1, input2):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_names = [inp.name for inp in session.get_inputs()]
    print("Input names:", input_names)
    inputs = {
        input_names[0]: input1,
        input_names[1]: input2,
    }
    
    outputs = session.run(None, inputs)
    for i, out in enumerate(outputs):
        print(f"Output {i}: shape={out.shape}")
    return outputs


if __name__ == "__main__":
    SIZE = 256
    RGB_path = cv2.imread('./data/real/color/1.jpg')
    RGB_img = cv2.resize(RGB_path, (SIZE, SIZE))
    RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    RGB_img = np.transpose(RGB_img, (2, 0, 1))  # 3*256*256, RGB
    RGB_img = torch.from_numpy(RGB_img).float()  # Convert to float tensor
    RGB_img.div_(255).sub_(0.5).div_(0.5)
    RGB_img = torch.unsqueeze(RGB_img, 0)

    IR_path = cv2.imread('./data/real/ir/1.jpg')
    IR_img = cv2.resize(IR_path, (SIZE, SIZE))
    IR_img = cv2.cvtColor(IR_img, cv2.COLOR_BGR2RGB)
    IR_img = np.transpose(IR_img, (2, 0, 1))  # 3*256*256, IR
    IR_img = torch.from_numpy(IR_img).float()  # Convert to float tensor
    IR_img.div_(255).sub_(0.5).div_(0.5)
    IR_img = torch.unsqueeze(IR_img, 0)

    # run pt
    pt_model = './saved/FASNet-epoch10.tar'
    weight = torch.load(pt_model, weights_only=True)
    net = ViT_AvgPool_2modal_CrossAtten_Channel()
    net.load_state_dict(weight, strict=True)
    net.eval()
    pt_feat, R, I = net(RGB_img, IR_img)

    # run onnx
    onnx_path = './saved/FASNet-epoch10.onnx'
    RGB_img = RGB_img.numpy()  # Convert to numpy array
    IR_img = IR_img.numpy()  # Convert to numpy array
    onnx_feat, R, I = onnx_run(onnx_path, RGB_img, IR_img)

    # See Mean squared Error
    print("PT output shape:", pt_feat)
    print("ONNX output shape:", onnx_feat)
