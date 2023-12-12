from PIL import Image;
# from sklearn import accuracy_score;
import cv2 as cv;
import numpy as np;
import os;

faceDetector = cv.face.LBPHFaceRecognizer.create();
faceDetector.read("./assets/Datasets/yalefaces/lbph_classifier.yml");

listExpecteds = [];
listPredicts = [];

paths = [
    os.path.join("./assets/Datasets/yalefaces/test/", f) 
    for f in os.listdir("./assets/Datasets/yalefaces/test/")
];

for path in paths:
    imageTest = Image.open(path);
    imageTestNumpy = np.array(imageTest, "uint8");

    predict = faceDetector.predict(imageTestNumpy);
    expected = int(os.path.split(path)[1].split(".")[0].replace("subject", ""));
    
    listExpecteds.append(expected);
    listPredicts.append(predict[0]);

listExpecteds = np.array(listExpecteds);
listPredicts = np.array(listPredicts);

# print([type(listExpecteds), type(listPredicts)]);
# print(listExpecteds);
# print(listPredicts);

result = [];
correct = 0;
incorrect = 0;

for i in range(0, len(listPredicts)):
    if listExpecteds[i] == listPredicts[i]:
        result.append([f"Previsao {listPredicts[i]} correta para a imagem {listExpecteds[i]}."]);
        correct+=1;
    else:
        result.append([f"Previsao {listPredicts[i]} incorreta para a imagem {listExpecteds[i]}."]);
        incorrect+=1;

for r in result:
    print(r);

print(f"Acertos: {correct} || Erros: {incorrect}");