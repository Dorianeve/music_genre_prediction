# Music Genre Prediction
## Machine Learning Assignment - MANILI

The project consists in a classification problem of a specific music genre given provided music features. The dataset is taken from this Kaggle repository - https://www.kaggle.com/insiyeah/musicfeatures which provided the following description: "the dataset provided here which consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format." The audio features were extracted via a python library called librosa utilising the code present at this link https://librosa.org/doc/latest/index.html. After trying different Classification algorithms and evaluating their performance, a sample of new songs, with features extrapolated via the same code and library, will be submitted to the machine learning best performing algorithm, to assign a "genre" to my music of interest.


## Problem statement
Music genres are established conventions, but probably they might be distiguish-able via some music features. Indeed, as genres are many, and very often they are overlapping, it is expected to encounter some challenges to have high prediction performances. The project has been also conducted only exclusively on two better differentiated genres (pop and classical), and the performance as expected were much higher.
The problem in a multi-class classification problem. The aim is to explore the performance of the faesible classification algoritms and evaluate the performances on 10 different music genre. Classifiers that are suitable for multi-class classification will be used.

## Project overview
The assignment is divided into steps:

- data exploration and preparation
- I algorithm: Support Vector Classifier
- II algorithm: Nearest Neighbor
- III algorithm: Decision Tree
- IV algorithm: Ensamble Methods (Random Forest / Extra Trees plus Boosting)
- Analysis results and findings
- Extraction of music features from selected music
- Prediction of extra music

# Data Exploration and Prparation
Dataset consists in 10,000 entries with 100 entries per class/label. The classes are balanced and equally represented: jazz, blues, disco, classical, pop, reggae, metal, hiphop, country, rock.
The dataset has 30 columns of which: 1 is the label column, and 28 the the "features" columns.
The features are all continuous variables, extracted with the python library "LIBROSA".
These are:


The features analysis suggests that they are almost normally distributed, which suggests the utilization of the StandardScaler() at a later stage, before launching the training algoritms. There are no categorical features to be encoded.
Performing a correlation analysis, it seems that some of the features are highly positively or negatively correlated.
These are the highly correlated features:



It is decided then to drop 'tempo', 'spectral centroid', 'mfcc2' and 'mfcc8' features, to reduce the correlation between features therefore not to undermine the performance of algoritms sensitive to correlation (such as RandomForest). 
