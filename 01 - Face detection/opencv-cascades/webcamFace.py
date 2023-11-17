import cv2 as cv;

faceDetector = cv.CascadeClassifier("./assets/Cascades/haarcascade_frontalface_default.xml");

video = cv.VideoCapture(0);

while True:
    ok, frame = video.read();
    imageGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY);

    facesDetected = faceDetector.detectMultiScale(
        imageGray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150)
    );

    for face in facesDetected:
        x, y, w, h = face;
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2);
        #print(f"Width: {w} ;; Height: {h}");
    
    cv.imshow("Video", frame);

    if cv.waitKey(1) == ord("q"):
        break;

video.release();
cv.destroyAllWindows();