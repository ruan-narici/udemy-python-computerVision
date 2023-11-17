from PIL import Image;
import cv2 as cv;
import numpy as np;
import os;

path = "./assets/Datasets/yalefaces/train/";
# print(os.listdir(path=path));

def getImageData():
    paths = [
        os.path.join(path, f)
        for f in os.listdir(path=path)
    ];

    # print(paths);
    faces = [];
    ids = [];

    for p in paths:
        # print(p);
        imagem = Image.open(p).convert('L');
        # print(type(imagem));
        imagemNumpy = np.array(imagem, 'uint8');
        # print(type(imagemNumpy));
        id = int(p.split(path)[1].split('.')[0].replace('subject', ''));
        faces.append(imagemNumpy);
        ids.append(id);
    return np.array(ids), faces;

ids, faces = getImageData();
