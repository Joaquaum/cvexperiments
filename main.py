import numpy as np
import cv2 as cv
import math

# Configurações
cap = cv.VideoCapture('video.mp4')
contador_garrafa = 0
lim_esquerda = 800   
distancia_maxima = 150


centroides_anteriores = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_bound = np.array([109, 151, 128])
    upper_bound = np.array([122, 255, 186])
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv.findContours(mask_clean, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    centroides_atuais = []
    cv.line(frame, (lim_esquerda, 0), (lim_esquerda, frame.shape[0]), (255, 0, 0), 2) #linha limite para as garrafas
    for c in contours:
        area = cv.contourArea(c)
        if 2900 < area < 4700:
            M = cv.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) 
                centroides_atuais.append((cx, cy))
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    if len(centroides_anteriores) > 0:
        for (cx, cy) in centroides_atuais:
            ponto_mais_proximo = None
            menor_distancia = float('inf')

            for (px, py) in centroides_anteriores:
                dist = math.hypot(cx - px, cy - py)
                
                if dist < menor_distancia:
                    menor_distancia = dist
                    ponto_mais_proximo = (px, py)

            if menor_distancia < distancia_maxima:
                px, py = ponto_mais_proximo

                if px > lim_esquerda and cx <= lim_esquerda:
                    contador_garrafa += 1
                    print("Contador de garrafas:" + str(contador_garrafa))
                    cv.line(frame, (lim_esquerda, 0), (lim_esquerda, frame.shape[0]), (0, 255, 255), 4)

    centroides_anteriores = centroides_atuais.copy()

    cv.putText(frame, f'Contagem: {contador_garrafa}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.imshow("Video", frame)
    # cv.imshow("Mascara", mask_clean)

    if cv.waitKey(30) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()