# Music Genre Prediction
## Machine Learning Assignment - MANILI

The project consists in a classification problem of a specific music genre given provided music features. The dataset is taken from this Kaggle repository - https://www.kaggle.com/insiyeah/musicfeatures which provided the following description: "the dataset provided here which consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format." The audio features were extracted via a python library called librosa utilising the code present at this link https://librosa.org/doc/latest/index.html. After trying different Classification algorithms and evaluating their performance, a sample of new songs, with features extrapolated via the same code and library, will be submitted to the machine learning best performing algorithm, to assign a "genre" to my music of interest.

## Project overview
The assignment is divided into steps:

data exploration and preparation
I algorithm: Support Vector Classifier
II algorithm: Nearest Neighbor
III algorithm: Decision Tree
IV algorithm: Ensamble Methods (Random Forest / Extra Trees plus Boosting)
Analysis results and findings
Extraction of music features from selected music
Prediction of extra music

## Problem statement
Music genres are established conventions, but probably they might be distiguish-able via some music features. Indeed, as genres are many, and very often they are overlapping, it is expected to encounter some challenges to have high prediction performances. The project has been also conducted only exclusively on two better differentiated genres (pop and classical), and the performance as expected were much higher.
