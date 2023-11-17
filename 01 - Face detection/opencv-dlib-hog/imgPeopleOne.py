import cv2 as cv;
import dlib as dlib;

faceDetector = dlib.get_frontal_face_detector();

image = cv.imread("./assets/Images/people1.jpg");
image = cv.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)));

facesDetected = faceDetector(image, 1);

for face in facesDetected:
    top, right, left, bottom = face.top(), face.right(), face.left(), face.bottom();
    cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3);

cv.imshow("Imagem", image);
cv.waitKey(0);
