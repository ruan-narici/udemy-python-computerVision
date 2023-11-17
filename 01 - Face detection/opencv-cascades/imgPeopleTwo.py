import cv2 as cv #Open CV

# Importando o modelo xml Cascade já treinado para deteccao de faces.
faceDetector = cv.CascadeClassifier('./assets/Cascades/haarcascade_frontalface_default.xml');

# Carregando a imagem e atribuindo a uma variavel
imageTwo = cv.imread("./assets/Images/people2.jpg");

# Diminuindo o tamanho da altura e lagura.
# imageTwo = cv.resize(imageTwo, (800, 600));

# Convertendo a image para canais de cinza
imageTwoGray = cv.cvtColor(imageTwo, cv.COLOR_BGR2GRAY);

# Executando a deteccao de faces através do metodo detectMultiScale
detections = faceDetector.detectMultiScale(image=imageTwoGray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32), maxSize=(100, 100));

# Exibindo as deteccoes
# Padrao de exibicao (x, y, width, height)

for x, y, w, h in detections:
    print([f"Width: {w}", f"Height: {h}"]);

# Percorrendo as deteccoes e criando retangulos na imagem original(colorida)
for x, y, w, h in detections: 
    # rectangle recebe como parametro a imagem, o inicio (x, y), o tamanho (x+width, y+height), a cor BGR, o tamanho do traço
    cv.rectangle(imageTwo, (x, y), (x + w, y + h), (0, 0, 255), 2);

cv.imshow('Faces detectadas', imageTwo);
cv.waitKey(0);