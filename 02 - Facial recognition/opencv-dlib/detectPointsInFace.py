import cv2 as cv;
import dlib;

faceDetector = dlib.get_frontal_face_detector();
pointDetector = dlib.shape_predictor("./assets/Weights/shape_predictor_68_face_landmarks.dat");

image = cv.imread("./assets/Images/people2.jpg");
facesDetected = faceDetector(image, 1);

for face in facesDetected:
    left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom();
    cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2);
    pointsDetected = pointDetector(image, face);
    for point in pointsDetected.parts():
        cv.circle(image, (point.x, point.y), 2, (0, 0, 255));

cv.imshow("Image", image);
cv.waitKey(0);