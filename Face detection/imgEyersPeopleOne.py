import cv2 as cv;

faceDetector = cv.CascadeClassifier("./assets/Cascades/haarcascade_frontalface_default.xml");
eyersDetector = cv.CascadeClassifier("./assets/Cascades/haarcascade_eye.xml");

image = cv.imread("./assets/Images/people1.jpg");
image = cv.resize(image, (int(image.shape[1] / 1.5) , int(image.shape[0] / 1.5)));

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);

facesDetected = faceDetector.detectMultiScale(imageGray, scaleFactor = 1.3);
eyersDetected = eyersDetector.detectMultiScale(
    imageGray, 
    scaleFactor = 1.04, 
    minNeighbors = 6,
    maxSize = (40, 40)
    );

for x, y, w, h in facesDetected:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3);

for x, y, w, h in eyersDetected:
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2);

cv.imshow("Image", image);
cv.waitKey(0);
