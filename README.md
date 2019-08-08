# Detect-Parkinsons

The researchers found that the drawing speed was slower and the pen pressure lower among Parkinson’s patients — this was especially pronounced for patients with a more acute/advanced forms of the disease.

Research paper link : https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full

One of the symptoms of Parkinson’s is tremors and rigidity in the muscles, making it harder to draw smooth spirals and waves.

Since the dataset is extremely small, applying deep learning here would not be the ideal choice.
So, I used HOG Image descriptor to quantify the images and then trained them using RandomForestClassifier on the top of extracted features.

The dataset used here was curated by Adriano de Oliveira Andrade and Joao Paulo Folado from the NIATS of Federal University of Uberlândia.
The dataset itself consists of 204 images and is pre-split into a training set and a testing set, consisting of:

Spiral: 102 images, 72 training, and 30 testing
Wave: 102 images, 72 training, and 30 testing

Modules Required :
1.  OpenCV
2.  NumPy
3.  Scikit-learn
4.  Scikit-image
5.  imutils



Should be run on the command line due to the usage of argsParser and the arguments required are --dataset and --trials.

E.g. :  python main.py --dataset dataset\spiral --trial 25 

Command should look something like this.

--dataset is a necessary argument

--trial is not a necessary argument. Default value of trial is 5.
