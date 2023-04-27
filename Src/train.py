import os
import random
import glob
import math
import itertools
import numpy as np
import pandas as pd
import albumentations
import cv2
import PotatoDataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' 

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(seed=42)


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
        return X


def train_fn(model,
             train_dataloader,
             optimizer,
             scheduler,
             loss_fn=None, 
             device=None,
             gradient_accumulation_steps=1, 
             grad_clip=0.0):
    
    model.train()
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    train_loss = 0.0
    
    for batch_idx, (image, label) in enumerate(train_dataloader):
        image, label = image.to(device), label.to(device)
        with ctx:
            output = model(image)
            loss = loss_fn(output, label)
            
        if batch_idx % gradient_accumulation_steps==0:
            loss = loss / gradient_accumulation_steps
        
        train_loss +=loss.item()
            
        scaler.scale(loss).backward()
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
    return train_loss/len(train_dataloader)
    
    
def evaluate_fn(model, valid_dataloader, loss_fn=None, device=None):
    
    model.eval()
    
    predictions = []
    valid_loss = 0.0
    
    for image, label in valid_dataloader:
        image, label = image.to(device), label.to(device)
        
        with ctx:
            output = model(image)
            loss = loss_fn(output, label)
        
        valid_loss +=loss.item()
        predictions.append(torch.argmax(output, axis=1).squeeze().cpu().detach().numpy())
        
    return valid_loss/len(valid_dataloader), np.concatenate(predictions)
    
    
def run_model(data, epochs=10, learning_rate=1e-4, batch_size=8, weight_decay=1e-5, seed=42):    
    
    seed_everything(seed)
    
    best_score = float(-np.inf)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    targets = data[CLASS_NAMES].values
    x_train, x_test, y_train, y_test = train_test_split(data['images'], targets, test_size=0.20, random_state=42)
    
    train_dataset = PotatoDataset.ClassificationDataset(
            image_paths=x_train.values,
            targets=y_train,
            resize=(256,256),
            augmentations=None
        )
    
    valid_dataset = PotatoDataset.ClassificationDataset(
            image_paths=x_test.values,
            targets=y_test,
            resize=(256,256),
            augmentations=None
        )

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = AlexNet()
    model.to(device)
     
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    
    for epoch in range(epochs): 
        
        train_loss = train_fn(model, train_loader, optimizer, scheduler, loss_fn=criterion, device=device)
        valid_loss, valid_preds = evaluate_fn(model, valid_loader, loss_fn=criterion, device=device)

        valid_score = f1_score(np.argmax(y_test, 1), valid_preds, average="micro")
        
        print(f'EPOCHS: {epoch}/{epochs}, valid metric_score: {valid_score:.4f}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}')
        
        if valid_score>best_score:
            best_score = valid_score
            print(f'best score is : {best_score}')
            torch.save(model.state_dict(), 'cnn_model_potato.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--path", type=str)
    
    args = parser.parse_args()
    
    CLASS_NAMES=['Potato___healthy','Potato___Late_blight','Potato___Early_blight']
    PATH = args.path 
    
    files = [[(file, idx) for file in glob.glob(os.path.join(PATH, cls+'\*.jpg'))] for idx, cls in enumerate(CLASS_NAMES)]
    potatoes_lst = list(itertools.chain(*files))
    
    df = pd.DataFrame(potatoes_lst, columns=['images', 'targets'])
    
    df[CLASS_NAMES] = np.eye(max(df.targets.values)+1)[df.targets.values]
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    run_model(df, 
              epochs=args.epochs,
              learning_rate=args.learning_rate, 
              batch_size=args.batch_size,
              weight_decay=args.weight_decay,
              seed=args.seed)
    
