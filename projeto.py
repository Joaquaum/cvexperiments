import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename == "CLASSE A":
            break
        category = []
        path = folder + "/" + filename
        for cat in sorted(os.listdir(path), key=natural_key):
            img = cv.imread(path + "/" + cat)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

images = load_images_from_folder(r'C:\Users\ppgmcs\Desktop\imagens tratadas - final')

def artefatos_sift(images):
    sift_vectors = {}
    descritor_lista = []
    sift = cv.SIFT_create()
    for k, value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img, None)
            descritor_lista.extend(des)
            features.append(des)
        sift_vectors[k] = features
    return [descritor_lista, sift_vectors]

sifts = artefatos_sift(images)
lista_descritores = sifts[0]
lista_features = sifts[1]

kmeans = KMeans(n_clusters=200, n_init= 10)
lista_descritores = np.array(lista_descritores, dtype=np.float64)
kmeans.fit(lista_descritores)
labels = kmeans.labels_

contador = 0 
dic_hist = {}
for classe, imagens in lista_features.items(): #cria uma lista com os histogramas para todas as imagens
    dic_hist[classe] = {}
    for idx, des_imagem in enumerate(imagens):
        nmr_des_imagem = len(des_imagem)
        labels_imagem = []
        
        for _ in range(nmr_des_imagem):
            labels_imagem.append(labels[contador])
            contador += 1
        
        histogram, _ = np.histogram(labels_imagem, bins=range(201))  
        
        dic_hist[classe][idx] = histogram

lista_svm = []
lista_labels_svm = []
for k, value in dic_hist.items():
    for hist in value.values():
        lista_svm.append(hist)
        if(k == "CLASSE 1.1"):
            lista_labels_svm.append(0)
        elif(k == "CLASSE 1.2"):
            lista_labels_svm.append(1)
        elif(k == "CLASSE 2.1"):
            lista_labels_svm.append(2)
        elif(k == "CLASSE 2.2"):
            lista_labels_svm.append(3)
model = SVC(kernel='linear', C=1.0)
model.fit(lista_svm, lista_labels_svm)
img2 = cv.imread(r"C:\Users\ppgmcs\Desktop\teste\imagem_38 classe 2.1.jpg")
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(img2, None)

if des is not None:
    labels = kmeans.predict(des.astype(np.float64))
    histograma = np.histogram(labels, bins=range(201))[0]
    pred = model.predict([histograma.astype(np.float64)])
    print("Predição:", pred)
else:
    print("Nenhum descritor encontrado.")


