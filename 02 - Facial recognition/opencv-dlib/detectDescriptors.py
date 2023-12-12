from PIL import Image;
import cv2 as cv;
import numpy as np;
import dlib;
import os;

faceDetector = dlib.get_frontal_face_detector();
pointDetector = dlib.shape_predictor("./assets/Weights/shape_predictor_68_face_landmarks.dat");
faceDescriptorExtractor = dlib.face_recognition_model_v1("./assets/Weights/dlib_face_recognition_resnet_model_v1.dat");

index = {};
idx = 0;
facesDescriptors = None;

paths = [
    os.path.join("./assets/Datasets/yalefaces/train/", f) 
    for f in os.listdir("./assets/Datasets/yalefaces/train/")
];

for path in paths:
    image = Image.open(path).convert("RGB");
    imageNumpy = np.array(image, "uint8");
    facesDetected = faceDetector(imageNumpy, 1);
    for face in facesDetected:
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom();
        cv.rectangle(imageNumpy, (left, top), (right, bottom), (0, 255, 0), 2);

        pointsDetected = pointDetector(imageNumpy, face);
        for point in pointsDetected.parts():
            cv.circle(imageNumpy, (point.x, point.y), 2, (0, 0, 255));
    
    faceDescriptor = faceDescriptorExtractor.compute_face_descriptor(imageNumpy, pointsDetected);
    ## Adicionando em formato de lista
    faceDescriptor = [f for f in faceDescriptor];
    ## Adicionando em formato array numpy
    faceDescriptor = np.asarray(faceDescriptor, dtype=np.float64);
    faceDescriptor = faceDescriptor[np.newaxis, :];

    ## Alimentando a variavel facesDescriptors
    if facesDescriptors is None:
        facesDescriptors = faceDescriptor;
    else:
        facesDescriptors = np.concatenate((facesDescriptors, faceDescriptor), axis=0);

    index [idx] = path;
    idx+=1;

faceDiference = np.linalg.norm(facesDescriptors[1] - facesDescriptors[9]);
print(faceDiference);
