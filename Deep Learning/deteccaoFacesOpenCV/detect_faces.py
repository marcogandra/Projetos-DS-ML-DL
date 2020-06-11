# Pacotes necessários
import numpy as np
import argparse
import cv2

# Construindo o argumento "parse" e passando os argumentos
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--imagem', required=True, help='caminho para a imagem')
ap.add_argument('-p', '--prototxt', required=True, help="caminho  para o arquivo Caffe prototxt")
ap.add_argument('-m', '--modelo', required=True, help='caminho para o modelo Caffe pré-treinado')
ap.add_argument('-c', '--confidencia', type=float, default=0.5, help='probabilidade mínima para detecções fracas')
args = vars(ap.parse_args())

# Carregando o model 
print('CARREGANDO O MODELO...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['modelo'])

# Carregando a imagem de entrada e construindo uma entrada do tipo 'blob' para a imagem,
# redimensionando para 300x300 pixels e normalizando
imagem = cv2.imread(args['imagem'])
(h, w) = imagem.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Passando o 'blob' através da rede e obtendo as detecções e predições
print('COMPUTANDO AS DETECÇÕES DE OBJETOS')
net.setInput(blob)
deteccoes = net.forward()

# Iterando através das detecções
for i in range(0, deteccoes.shape[2]):
    # Extraindo a confidência associada em cada predição
    confidencia = deteccoes[0, 0, i, 2]

    # Filtrando apenas as detecções com confidência maior que a confidência
    # mínima
    if confidencia > args['confidencia']:
        # Computando as coordenadas (x, y) para a 'box' do objeto
        box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # Desenhando a 'box' e a possibilidade através das faces detectadas
        texto = '{:.2f}%'.format(confidencia * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(imagem, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(imagem, texto, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Mostrando a imagem de saída
cv2.imshow('Saida', imagem)
cv2.waitKey(0)