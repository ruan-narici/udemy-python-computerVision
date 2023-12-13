from PIL import Image;
import cv2 as cv;
import numpy as np;
import dlib;
import os;

paths = [os.path.join("./assets/Datasets/yalefaces/train/", f) 
        for f in os.listdir("./assets/Datasets/yalefaces/train/")
];

index = {};
idx = 0;
facesDescriptors = None;

faceDetector = dlib.get_frontal_face_detector();
pointFaceDetector = dlib.shape_predictor("./assets/Weights/shape_predictor_68_face_landmarks.dat");
faceDescriptorExtractor = dlib.face_recognition_model_v1("./assets/Weights/dlib_face_recognition_resnet_model_v1.dat");

for path in paths:
    #print(path);
    image = Image.open(path).convert("RGB");
    image_np = np.array(image, "uint8");
    #print(type(image_np));
    facesDetected = faceDetector(image_np, 1);
    for face in facesDetected:
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom();
        cv.rectangle(image_np, (left, top), (right, bottom), (255, 0, 0), 2);
    #cv.imshow("Image", image_np);
    #cv.waitKey(0);
        facesPointersDetecteds = pointFaceDetector(image_np, face);
        for pointers in facesPointersDetecteds.parts():
            cv.circle(image_np, (pointers.x, pointers.y), 2, (0, 0, 255));
    
    faceDescriptor = faceDescriptorExtractor.compute_face_descriptor(image_np, facesPointersDetecteds);
    ## Adicionando em formato de lista
    faceDescriptor = [f for f in faceDescriptor];
    ## Adicionando em formato array numpy
    faceDescriptor = np.asanyarray(faceDescriptor, dtype=np.float64);
    faceDescriptor = faceDescriptor[np.newaxis, :];
    
    ## Alimentando a variavel facesDescriptors
    if facesDescriptors is None:
        facesDescriptors = faceDescriptor;
    else:
        facesDescriptors = np.concatenate((facesDescriptors, faceDescriptor), axis=0);
    index [idx] = path;
    idx+=1;

    distances = np.linalg.norm(faces)