import onnxruntime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.ViT import ViT_AvgPool_2modal_CrossAtten_Channel


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def onnx_run(rgb_data, ir_data, onnx_model):
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(onnx_model, so)
    
    def to_numpy(tensor):
        if tensor.requires_grad:
            np_tensor = tensor.detach().cpu().numpy()
        else:
            np_tensor = tensor.cpu().numpy()
        return np_tensor
    
    ort_inputs = {'R_data': to_numpy(rgb_data),
                  'I_data': to_numpy(ir_data)}
    
    pred, _, _ = ort_session.run(None, ort_inputs)
    return pred


if __name__ == "__main__":
    threshold = 0.99977154

    SIZE = 256
    RGB_img = cv2.imread('./data/fake/color/11.jpg')
    RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    RGB_img = cv2.resize(RGB_img, (SIZE, SIZE))
    RGB_img = np.transpose(RGB_img, (2, 0, 1))  # 3*256*256, RGB
    RGB_img = torch.Tensor(RGB_img).cuda()
    RGB_img = NormalizeData_torch(RGB_img)
    RGB_img = torch.unsqueeze(RGB_img, 0)

    IR_img = cv2.imread('./data/fake/ir/11.jpg')
    IR_img = cv2.cvtColor(IR_img, cv2.COLOR_BGR2RGB)
    IR_img = cv2.resize(IR_img, (SIZE, SIZE))
    IR_img = np.transpose(IR_img, (2, 0, 1))  # 3*256*256, IR
    IR_img = torch.Tensor(IR_img).cuda()
    IR_img = NormalizeData_torch(IR_img)
    IR_img = torch.unsqueeze(IR_img, 0)
    IR_img = TF.rgb_to_grayscale(IR_img, num_output_channels=3)
    print(RGB_img.shape, IR_img.shape)

    # run pt
    device_id = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pt_model = './saved/FASNet-epoch10.tar'
    weight = torch.load(pt_model, weights_only=True)
    net = ViT_AvgPool_2modal_CrossAtten_Channel().to(device_id)
    net.load_state_dict(weight, strict=True)
    net.eval()
    pred, R, I = net(RGB_img, IR_img)
    score = F.softmax(pred, dim=1).cpu().data.numpy()[:, 1]  # multi class
    print(score)

    # run onnx
    onnx_path = './saved/FASNet-epoch10.onnx'
    pred = onnx_run(RGB_img, IR_img, onnx_path)[0]
    pred = torch.tensor(pred, dtype=torch.float32)

    score = F.softmax(pred)[1].item()
    print(score)
