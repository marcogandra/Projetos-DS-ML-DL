# Importando as bibliotecas necessarias
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Criando os argumentos
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=True, 
                help='caminho para o arquivo Caffe prototxt')
ap.add_argument('-m', '--modelo', required=True, 
                help='caminho para o modelo Caffe pre-treinado')
ap.add_argument('-c', '--confidencia', type=float, default=0.5,  
                help='probabilidade minima para filtrar deteccoes fracas')
args = vars(ap.parse_args())

# Carregando o modelo
print('CARREGANDO O MODELO...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['modelo'])

# Inicializando o video
print('INICIALIZANDO VIDEO...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Iterando atraves dos 'frames' do video
while True:
    # Pegando os frames e redimensionando para ter largura
    # maxima de 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Pegando as dimensoes do video e convertendo em um 'blob'
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Passando o 'blob' atraves da rede e obtendo as deteccoes e predicoes
    net.setInput(blob)
    deteccoes = net.forward()

    # Iterando atraves das deteccoes
    for i in range(0, deteccoes.shape[2]):
        # Extrando a confidencia de cada deteccao
        confidencia = deteccoes[0, 0, i, 2]

        # Filtrando as deteccoes onde a confidencia e maior que a confidencia minima
        if confidencia < args['confidencia']:
            continue

        # Computando as coordenadas (x, y)
        box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # Desenhando a box atraves das deteccoes juntamente com a probabilidade
        texto = '{:.2f}%'.format(confidencia * 100)
        y = startY - 10 if startY - 10 > 0 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, texto, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Exibindo as saidas
    cv2.imshow('Frame', frame)
    botao = cv2.waitKey(1) & 0xFF

    # Se o botao 'q' for pressionado, para o loop
    if botao == ord('q'):
        break
# Destroi as janelas e para a filmagem 
cv2.destroyAllWindows()
vs.stop()