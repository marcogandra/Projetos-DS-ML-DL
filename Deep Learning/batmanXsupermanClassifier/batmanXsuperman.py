# Importando as bibliotecas necessárias
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Classe que contém toda a interface do programa
class Interface():
    
    # Função para selecionar e carregador no programa a imagem que será classificada
    def iapture(self):
     
        self.filename = askopenfilename()

        self.image = Image.open(self.filename)

        self.photo = ImageTk.PhotoImage(self.image)

        label = Label(self.root, image=self.photo).grid(row=1, column=0, padx=15, pady=5, rowspan=3)
        label.image = self.photo

    # Função para treinar a rede neural com as imagens da pasta batmanXsuperman
    def train(self):
        
        # Criando o modelo da rede
        self.classifier = Sequential()
        
        # Adicionando as convoluções
        self.classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Adicionando a camada de flatten
        self.classifier.add(Flatten())

        # Criando a rede neural densa
        self.classifier.add(Dense(units=128, activation='relu'))
        self.classifier.add(Dense(units=128, activation='relu'))
        # Criando a camada de saída
        self.classifier.add(Dense(units=1, activation='sigmoid'))

        # Compilando a rede neural
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy',
                              metrics=['accuracy'])
        
        # Criando um gerador de imagens para treinamento
        train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                                 rotation_range=7,
                                                 horizontal_flip=True,
                                                 shear_range=0.2,
                                                 height_shift_range=0.07,
                                                 zoom_range=0.2)

        # Criando um gerador de imagens para teste
         test_data_gen = ImageDataGenerator(rescale=1. / 255)

        # Usando o gerador de imagens de treinamento para criar a base de treinamento
        train_base = train_data_gen.flow_from_directory('batmanXsuperman/training_set',
                                                            target_size = (64,64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')
    
    
    
        # Usando o gerador de imagens de teste para criar a base de teste
        test_base = test_data_gen.flow_from_directory('batmanXsuperman/test_set',
                                                            target_size = (64,64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')
        
        # Treinando o classificador. Não utilizei muitas épocas devido a capacidade de processamento 
        # do meu PC não ser tão grande
        self.classifier.fit_generator(train_base, steps_per_epoch=5000/32,
                                    epochs=3, validation_data=test_base,
                                    validation_steps=370/32)

        self.counter = 1

    # Função para classificar a imagem
    def classifyImage(self):
        
        # Carregando a imagem em uma variável
        test_image = image.load_img(self.filename,
                                   target_size = (64, 64))
          
        # Transformando a imagem em array 
        test_image = image.img_to_array(test_image)
        # Deixando o array na escala entre 0 e 1
        test_image /= 255
        
        # Expandindo a dimensão no array para ser aceitável no classificador
        test_image = np.expand_dims(imagem_teste, axis= 0)
        
        # Usando o classificador para prever a imagem
        pred = self.classificador.predict(test_image)
        
        # Se o classificador prever 0 (e já que 0 é menor que 0.5) é uma imagem do Batman
        if pred <= 0.5:
            print('A imagem é do Batman')
            Label(self.root, text='É O BATMAN').grid(row=1, column=0)
            
        # Se o classificador prever 1 (e já que 1 é maior que 0.5) é uma imagem do Superman
        elif pred >=0.5:
            print('A imagem é do Superman')
            Label(self.root, text='É O SUPERMAN').grid(row=1, column=0)
            

    def __init__(self):
        self.counter = 0
        self.root = Tk()
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW")
        self.root.title("Identificaçao de imagens")
        
        # Criando o botão para selecionar uma imagem
        Button(self.root,text='Selecione a imagem', command=self.Captura).grid(row=0, column=0, pady=5)
        
        # Criando o botão para treinar a rede
        Button(self.root, text='Treinar Rede', command=self.Treinamento, width=10, height=2).grid(row=0, column=1)
    
        # Criando o botão para classificar a imagem
        Button(self.root,text='Classificar', command=self.ClassificarImagens, width=10, height=2).grid(row=1, column=1)

        self.root.mainloop()


Interface()
