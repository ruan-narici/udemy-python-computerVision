from PIL import Image;
import cv2 as cv;
import numpy as np;
import os;

path = "./assets/Datasets/yalefaces/train/";
# print(os.listdir(path));

def getImageData():
    ids = [];
    faces = [];

    paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
    ];
    # print(paths);

    for p in paths:
        image = Image.open(p).convert('L');
        imageNumpy = np.array(image, 'uint8');
        
        id = int(p.split(path)[1].split('.')[0].split('subject')[1]);
        face = imageNumpy;
        
        ids.append(id);
        faces.append(face);
    
    return np.array(ids), faces;

ids, faces = getImageData();

lbphClassifier = cv.face.LBPHFaceRecognizer.create();
lbphClassifier.train(faces, ids);
lbphClassifier.write("./assets/Datasets/yalefaces/lbph_classifier.yml");
