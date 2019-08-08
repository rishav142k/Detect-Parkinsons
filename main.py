from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
import os


def quantify(image) :
    features = feature.hog(image , orientations = 9 , pixels_per_cell =
    (8,8) , cells_per_block = (1,1) , transform_sqrt = True ,
    block_norm = "L1")

    return features



def load_split(path) :
    
    imagePaths = list(paths.list_images(path))
    data =  []
    labels = []

    for imagePath in imagePaths :
        label = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image , (200,200))

        image = cv2.threshold(image , 0,255, cv2.THRESH_BINARY_INV |
        cv2.THRESH_OTSU)[1]

        features = quantify(image)

        data.append(features)
        labels.append(label)

    return (np.array(data) , np.array(labels))
            

ap = argparse.ArgumentParser()
ap.add_argument("-d" , "--dataset" , required = True , help = "path to input dataset")
ap.add_argument("-t" , "--trials" , type = int , default = 5 , help = "no. of trials to run ")
args = vars(ap.parse_args())



trainingPath = os.path.sep.join([args["dataset"] , "training"])
testingPath = os.path.sep.join([args["dataset"] , "testing"])

print("[HOLD_ON] loading data....")
(trainX , trainY) = load_split(trainingPath)
(testX , testY) = load_split(testingPath)


le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

trials ={}


for i in range( 0 , args["trials"]) :
    print("[HOLD] training model {} of {}... ".format(i+1 , args["trials"]))
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(trainX , trainY)


    predicts = model.predict(testX)
    metrics = {}

    conf_matrix = confusion_matrix(testY , predicts).flatten()
    (tn , fp , fn , tp ) = conf_matrix
    metrics["Accuracy"] = (tp + tn)/float(conf_matrix.sum())
    metrics["Sensitivity"] =  tp/float(tp + tn)
    metrics["Specificity"] = tn / float(tp + tn)


    for (k,v) in metrics.items() :
        l =  trials.get(k , [])
        l.append(v)
        trials[k] = l


for metric in ("Accuracy" , "Sensitivity" , "Specificity") :

    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)


    print(metric)
    print("=" * len(metric))
    print("u = {:.4f} , o = {:.4f}" .format(mean , std))
    print("")


testingPaths = list(paths.list_images(testingPath))
indexes = np.arange(0 , len(testingPaths))
indexes = np.random.choice(indexes , size = (25, ) , replace = False)
images = []




for i in indexes :

    image = cv2.imread(testingPaths[i])
    output = image.copy()
    output = cv2.resize(output , (128,128))



    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image =cv2.resize(image , (200,200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    features = quantify(image)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0] 

    color = (0,255,0) if label == "healthy" else (0,0,255)
    cv2.putText(output , label, (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,
    color , 2)
    images.append(output)

montage = build_montages(images , (128,128),(5,5))[0]

cv2.imshow("Output" , montage)
cv2.waitKey(0)
