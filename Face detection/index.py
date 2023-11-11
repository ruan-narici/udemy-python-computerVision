import cv2 as cv #Open CV

image = cv.imread('./assets/Images/people1.jpg');
print(image.shape); # Lagura, altura e Canais (RGB)

# cv.imshow('image', image);
# cv.waitKey(0);

image = cv.resize(image, (800, 600)); # Diminuindo o tamanho da altura e lagura.

print(image.shape); # Lagura, altura e Canais (RGB) após o resize

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY); # Convertendo a image para canais de cinza

cv.imshow('image cinza', imageGray); # Exibindo a image
cv.waitKey(0); # O programa sera executado ate que uma tecla seja pressionada.
