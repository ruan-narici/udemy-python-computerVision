import cv2 as cv;

clockDetector = cv.CascadeClassifier("./assets/Cascades/clocks.xml");

image = cv.imread("./assets/Images/clock.jpg");
# image = cv.resize(image, (800, 600));

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);

clocksDetecteds = clockDetector.detectMultiScale(
    imageGray,
    scaleFactor=1.007,
    minNeighbors=5,
    maxSize=(105, 105)
);

for x, y, w, h in clocksDetecteds:
    print([w, h]);
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2);

cv.imshow("Image", image);
cv.waitKey(0);
