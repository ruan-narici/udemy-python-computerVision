import cv2 as cv;

fullBodyDetector = cv.CascadeClassifier("./assets/Cascades/fullbody.xml");

image = cv.imread("./assets/Images/people3.jpg");
image = cv.resize(image, (int(image.shape[1] * 2), int(image.shape[0] * 2)));

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);

bodysDetected = fullBodyDetector.detectMultiScale(
    imageGray,
    scaleFactor=1.005,
    minNeighbors=5,
    minSize=(110, 180)
);

for x, y, w, h in bodysDetected:
    print([w, h])
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2);

cv.imshow("Image", image);
cv.waitKey(0);
