import cv2 as cv;

carDetector = cv.CascadeClassifier("./assets/Cascades/cars.xml");

image = cv.imread("./assets/Images/car.jpg");

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);

carsDetected = carDetector.detectMultiScale(
    imageGray, 
    scaleFactor=1.007, 
    minNeighbors=3,
    minSize=(27, 27),
    maxSize=(58, 58)
    );

for x, y, w, h in carsDetected:
    print([w, h]);
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2);

cv.imshow("image", image);
cv.waitKey(0);

# imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);
