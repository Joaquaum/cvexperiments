import cv2 as cv
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from engine import train_one_epoch
from torchvision.ops import box_iou

def achar_bounding_boxes(imagem, linhas_yolo):
    coordenadas = []
    h, w, _ = imagem.shape
    
    for linha in linhas_yolo:
        valores = linha.strip().split()
        if len(valores) != 5:
            print(f"[ERRO] Formato inválido na linha: {linha}")
            continue

        classe, x_center, y_center, largura, altura = map(float, valores)

        x_center *= w
        y_center *= h
        largura *= w
        altura *= h

        x1 = int(x_center - largura / 2)
        y1 = int(y_center - altura / 2)
        x2 = int(x_center + largura / 2)
        y2 = int(y_center + altura / 2)
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1,y2)
        coordenadas.append((classe + 1, xmin, ymin, xmax, ymax))  
    return coordenadas

def criar_dic_txt(folder):
    dic_txt = {}
    for classe in os.listdir(folder):
        if classe == "CLASSE A":
            break
        categoria = []
        path = os.path.join(folder, classe)
        arquivos = sorted(os.listdir(path))
        for txt in arquivos:
            with open(os.path.join(path, txt), 'r')as f:
                linhas = f.readlines()
                categoria.append(linhas)
        dic_txt[classe] = categoria
    return dic_txt


def criar_imagens_dic(folder):
    dic_img = {}
    for classe in os.listdir(folder):
        if classe == "CLASSE A":
            break
        categoria = []
        path = os.path.join(folder, classe)
        arquivos = sorted(os.listdir(path))
        for img in arquivos:
            imagem = cv.imread(path + "/" + img)  
            categoria.append(imagem)
        dic_img[classe] = categoria
    return dic_img                  

def criar_dic_bounding_boxes(txts, imgs):
    dic_bb = {}
    for (classe, images), (classe2, txts) in zip(imgs.items(), txts.items()):
        categoria = []
        for (values, image), (values2, txt) in zip(enumerate(images), enumerate(txts)):
            categoria.append(achar_bounding_boxes(image, txt))
        dic_bb[classe] = categoria
    return dic_bb
    
txts = criar_dic_txt(r"C:\Users\ppgmcs\Desktop\aplicativo\txtimagens1")
imgs = criar_imagens_dic(r"C:\Users\ppgmcs\Desktop\aplicativo\imagens tratadas - final")
dic_bb = criar_dic_bounding_boxes(txts, imgs)


class MyDataset(Dataset):
    def __init__(self, imgs, dic_bb, transforms):
        self.imgs = imgs
        self.dic_bb = dic_bb
        self.transforms = transforms
        self.data = []
        for classe, imagem in imgs.items():
            for i, img in enumerate(imagem):
                bboxes = dic_bb[classe][i]
                if (len(bboxes) > 0):
                    self.data.append((img,bboxes))
    def __getitem__(self, idx):
        img, bboxes = self.data[idx]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        boxes = []
        labels = []
        for (classe, xmin, ymin, xmax, ymax) in bboxes:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(classe))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.data)



def trasformacao(treino):
    transforms = []
    if treino:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transforms)


dataset = MyDataset(imgs, dic_bb, trasformacao(True))

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 5
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
dl = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
dl_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr = 0.005,
    momentum = 0.9,
    weight_decay = 0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.1)
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
epocas = 2

@torch.no_grad()
def evaluation(model, data_loader, device):
    model.eval()
    total_iou = 0.0
    total_samples = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        for output, target in zip(outputs, targets):
            if len(output["boxes"]) == 0 or len(target["boxes"]) == 0:
                continue  # ignora se não houver previsão ou anotação

            ious = box_iou(output["boxes"], target["boxes"])  # [preds, gts]
            max_ious, _ = ious.max(dim=1)  # pega IoU da melhor correspondência
            total_iou += max_ious.mean().item()
            total_samples += 1
    mean_iou = total_iou / total_samples if total_samples > 0 else 0
    print(f"Validation IoU médio: {mean_iou:.4f}")
    return mean_iou

for e in range(epocas):
    train_one_epoch(model, optimizer, dl, device, e, print_freq=10)
    lr_scheduler.step()
    evaluation(model, dl_test, device)
    
