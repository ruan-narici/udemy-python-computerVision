from PIL import Image;
import cv2 as cv;
import numpy as np;
import os;

lbphFaceDetector = cv.face.LBPHFaceRecognizer().create();
lbphFaceDetector.read("./assets/Datasets/yalefaces/lbph_classifier.yml");

expectedPersonPath = "./assets/Datasets/yalefaces/test/subject01.happy.gif";

imagePerson = Image.open(expectedPersonPath).convert('L');
imagePersonNumpy = np.array(imagePerson, 'uint8');

predict = lbphFaceDetector.predict(imagePersonNumpy);

expectedId = int(os.path.split(expectedPersonPath)[1].split('.')[0].replace('subject', ''));

cv.putText(
    img=imagePersonNumpy, 
    text=f"Previsao: {predict[0]}", 
    org=(10, 30), 
    fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,
    fontScale=0.7,
    color=(0, 0, 0)
    );

cv.putText(
    img=imagePersonNumpy, 
    text=f"Esperado: {expectedId}", 
    org=(10, 50), 
    fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,
    fontScale=0.7,
    color=(0, 0, 0)
    );

cv.imshow("Image", imagePersonNumpy);
cv.waitKey(0);
