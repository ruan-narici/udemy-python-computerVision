import cv2 as cv;

bodyDetector = cv.CascadeClassifier("./assets/Cascades/fullbody.xml");

video = cv.VideoCapture(0);

humanDetected = False;
image = cv.imread("./assets/Images/alerta-1.png");
image = cv.resize(image, (800, 600));


while True:
    ok, frame = video.read();
    
    imageGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY);
    bodysDetected = bodyDetector.detectMultiScale(
        imageGray,
        scaleFactor=1.005,
        minNeighbors=3,
        minSize=(200, 350)
    );

    for body in bodysDetected:
        x, y, w, h = body;
        print(f"Width: {w} ;; Height: {h}");
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2);
        if humanDetected == False:
            cv.destroyAllWindows();
            cv.imshow("Alerta", image);
            humanDetected = True;
    
    if humanDetected == False:
        cv.imshow("Video", frame);
    
    if cv.waitKey(1) == ord("q"):
        break;

video.release();
cv.destroyAllWindows();

