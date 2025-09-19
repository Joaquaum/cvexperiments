import cv2 as cv
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

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
        coordenadas.append((classe, xmin, ymin, xmax, ymax))  
    return coordenadas

def criar_dic_txt(folder):
    dic_txt = {}
    for classe in os.listdir(folder):
        categoria = []
        path = os.path.join(folder, classe)
        for txt in os.listdir(path):
            with open(os.path.join(path, txt), 'r')as f:
                linhas = f.readlines()
                categoria.append(linhas)
        dic_txt[classe] = categoria
    return dic_txt


def criar_imagens_dic(folder):
    dic_img = {}
    for classe in os.listdir(folder):
        categoria = []
        path = os.path.join(folder, classe)
        for img in os.listdir(path):
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
    def __init__(self, imgs, dic_bb, transforms=None):
        self.imgs = imgs
        self.dic_bb = dic_bb
        self.transforms = transforms
        self.data = []
        for classe, imagem in imgs.items():
            for i, img in enumerate(imagem):
                bboxes = dic_bb[classe][i]
                self.data.append((img,bboxes))
    def __getitem__(self, idx):
        img, bboxes = self.data[idx]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        
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

dataset = MyDataset(imgs, dic_bb)
img, target = dataset[0]
print(img)
print(target)





        
        
        
    
