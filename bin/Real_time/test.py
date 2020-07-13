import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main(content_path,style_path,model_path,output_path):

    # set device on GPU if available, else CPU
    device = torch.device('cuda:0')

    # set model
    model = Model()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(content_path).convert("RGB")
    s = Image.open(style_path).convert("RGB")
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor)
    
    out = denorm(out, device)
    save_image(out, output_path, nrow=1)

if __name__ == '__main__':
    main("C:\\Users\\HP\\Desktop\\pku.jpg",
    "C:\\Users\\HP\\Desktop\\style.jpg",
    "C:\\Users\\HP\\Desktop\\model_state.pth",
    "C:\\Users\\HP\\Desktop\\target.jpg")