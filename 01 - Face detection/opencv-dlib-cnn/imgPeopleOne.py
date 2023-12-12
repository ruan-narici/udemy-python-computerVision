import cv2 as cv;
import dlib;

faceDetector = dlib.cnn_face_detection_model_v1("./assets/Weights/mmod_human_face_detector.dat");

image = cv.imread("./assets/Images/people2.jpg");
# image = cv.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)));

facesDetected = faceDetector(image, 1);

for face in facesDetected:
    left, top, right, bottom, confidence = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence;
    cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2);
    print(confidence);

cv.imshow("Image", image);
cv.waitKey(0);
