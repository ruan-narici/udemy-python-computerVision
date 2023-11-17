from PIL import Image;
import cv2 as cv;
import numpy as np;
import zipfile;

path = "./assets/Datasets/yalefaces.zip";
zipObject = zipfile.ZipFile(file=path, mode='r');
zipObject.extractall("./assets/Datasets/");
zipObject.close();
