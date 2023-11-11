import cv2 as cv #Open CV

image = cv.imread('./assets/Images/people1.jpg');

# Lagura, altura e Canais (RGB)
print(image.shape); 

# cv.imshow('image', image);
# cv.waitKey(0);

# Diminuindo o tamanho da altura e lagura.
image = cv.resize(image, (800, 600)); 

# Lagura, altura e Canais (RGB) após o resize
print(image.shape); 

# Convertendo a image para canais de cinza
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY); 

# Exibindo a image
# cv.imshow('image cinza', imageGray); 

# O programa sera executado ate que uma tecla seja pressionada.
# cv.waitKey(0); 

# Importando o modelo xml Cascade já treinado para deteccao de faces.
faceDetector = cv.CascadeClassifier('./assets/Cascades/haarcascade_frontalface_default.xml');

# Executando a deteccao de faces através do metodo detectMultiScale
detections = faceDetector.detectMultiScale(imageGray, 1.09);

# Exibindo as deteccoes
# Padrao de exibicao (x, y, width, height)
print(detections);

# Percorrendo as deteccoes e criando retangulos na imagem original(colorida)
for x, y, w, h in detections: 
    # rectangle recebe como parametro a imagem, o inicio (x, y), o tamanho (x+width, y+height), a cor BGR, o tamanho do traço
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2);

cv.imshow('Faces detectadas', image);
cv.waitKey(0);

