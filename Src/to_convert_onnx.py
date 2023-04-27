import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size = (11, 11), stride = (4, 4), padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size = (5, 5), stride = (1, 1), padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size = (3, 3), stride = (1, 1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size = (3, 3), stride = (1, 1), padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size = (3, 3), stride = (1, 1), padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2))
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=3)
    
    def forward(self, image):
        X = F.relu(self.conv1(image))
        X = self.max_pool(X)
        X = F.relu(self.conv2(X))
        X = self.max_pool(X)
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = F.relu(self.conv5(X))
        X = self.max_pool(X)
        X = self.dropout(X)
        X = X.view(-1, 256*6*6)
        X = self.linear1(X)
        X = self.dropout(X)
        X = self.linear2(X)
        return torch.softmax(X, 1)


def convert_onnx(input_path=None, onnx_path=None):
    model =AlexNet()
    model.load_state_dict(torch.load(input_path))
    model.to('cpu')
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    
    input_names = ["input1"]
    output_names = ["output1"]
    
    torch.onnx.export(
      model, 
      dummy_input, 
      onnx_path, 
      verbose=True, 
      input_names=input_names,
      output_names=output_names,
      opset_version=11
    )
	
	
if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--onnx_file_path", type=str)
    args = parser.parse_args()
    convert_onnx(args.input_path, args.onnx_file_path)