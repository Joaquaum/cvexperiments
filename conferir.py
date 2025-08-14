import cv2 as cv
import os
import re

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in sorted(os.listdir(path), key=natural_key):
            img = cv.imread(path + "/" + cat)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

def load_txt_from_folder(folder):
    txts = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in sorted(os.listdir(path), key = natural_key):
            with open(path + "/" + cat, 'r') as f:
                linhas = f.readlines()
                category.append(linhas)
        txts[filename] = category
    return txts

dicionario = load_images_from_folder('imagens tratadas - final')
dicionario_textos = load_txt_from_folder('C:/Users/ppgmcs/Desktop/aplicativo/txtimagens')
def desenhar_bounding_boxes(imagem, linhas_yolo, B, G, R):

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

        cv.rectangle(imagem, (x1, y1), (x2, y2), (B, G, R), 2)
    return imagem

for (classe, images), (classe2, txts) in zip(dicionario.items(), dicionario_textos.items()):
    for (values, image), (values2, txt) in zip(enumerate(images), enumerate(txts)):
        if (classe == "CLASSE 1.1"):
            img = desenhar_bounding_boxes(image, txt, 0, 255, 0)
        elif(classe == "CLASSE 1.2"):
            img = desenhar_bounding_boxes(image, txt, 255, 255, 0)
        elif(classe == "CLASSE 2.1"):
            img = desenhar_bounding_boxes(image, txt, 0, 255, 255)
        elif(classe == "CLASSE 2.2"):
            img = desenhar_bounding_boxes(image, txt, 255, 0, 0)
        elif(classe == "CLASSE A"):
            img = desenhar_bounding_boxes(image, txt, 0, 0, 255)
        path = "redesenho/" + classe
        if not os.path.exists(path):
            os.makedirs(path)

        cv.imwrite(path + "/imagem_" + str(values) + ".jpg", img)
        
        
        
    

