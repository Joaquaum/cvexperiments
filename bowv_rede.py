from torch import nn
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import cv2 as cv
from sklearn.cluster import KMeans
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os

#diciionario em python, um indice->uma lista



#transform = transforms.ToTensor()
#x = datasets.CIFAR10(root = "./data", train = True, transform=transform, download=True)
#teste = datasets.CIFAR10(root = "./data", train = False, transform = transform, download=True)


lista_labels = []
lista_descritores = []
#for i in range(len(teste)):

#    img, label = teste[i]
 #   path = os.path.join(r"C:\Users\ppgmcs\Desktop\louco\teste", str(label))
  #  os.makedirs(path, exist_ok=True)
   # arrImg = (img.numpy().transpose(1, 2, 0)* 255).astype('uint8')
    #cvImg = cv.cvtColor(arrImg, cv.COLOR_RGB2BGR)
    #cv.imwrite(path + "/"+ str(i) + ".jpg", cvImg)
contador_descritores = 0
sift = cv.SIFT_create()
pasta = r"C:\Users\ppgmcs\Desktop\louco\teste"
for classe in os.listdir(pasta):
    path = pasta + "/" + classe
    for img in os.listdir(path):
        img = cv.imread(path + "/" + img)
        kp, des = sift.detectAndCompute(img, None)
        if(contador_descritores == 30):
            break
        if des is not None:
            lista_descritores.append(des)
            lista_labels.append(int(classe))
            contador_descritores += 1
lista_descritores2 = np.vstack(lista_descritores)


kmeans = KMeans(n_clusters=50, n_init=10)
kmeans.fit(lista_descritores2)
labels = kmeans.labels_
contador = 0

lista_hist = []
for des in lista_descritores:
    nmr_des_imagem = len(des)
    labels_imagem = []
    for _ in range(nmr_des_imagem):
        labels_imagem.append(labels[contador])
        contador += 1
    histogram = np.histogram(labels_imagem, bins=range(51))
    lista_hist.append(histogram)
    
class rede(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(50, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.linear2(x)
        return x

lista_labels = torch.tensor(lista_labels)
tensores_hist = []
for i in lista_hist:
    tensores_hist.append(torch.tensor(i[0]))

model = rede()
criterio = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)
for i in range(500):
    model.train()
    for idx, n in enumerate(tensores_hist):
        n = n.float()
        y = model(n)
        loss = criterio(y, lista_labels[idx])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

img = cv.imread(r"C:\Users\ppgmcs\Desktop\louco\treino\0\29.jpg")
kp2, des2 = sift.detectAndCompute(img, None)
labels2 = kmeans.predict(des2)
histogram_teste = np.histogram(labels2, bins=range(51))
tensor_teste = torch.tensor(histogram_teste[0])
with torch.no_grad():
    print(model(tensor_teste.float()))
    print("aaaaaaaaaaaaaaaaaaaa")
    print(0)
