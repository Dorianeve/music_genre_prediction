# Music Genre Prediction
## Machine Learning Assignment - MANILI

The project consists in a classification problem of a specific music genre given provided music features. The dataset is taken from this Kaggle repository - https://www.kaggle.com/insiyeah/musicfeatures which provided the following description: "the dataset provided here which consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format." The audio features were extracted via a python library called librosa utilising the code present at this link https://librosa.org/doc/latest/index.html. After trying different Classification algorithms and evaluating their performance, a sample of new songs, with features extrapolated via the same code and library, will be submitted to the machine learning best performing algorithm, to assign a "genre" to my music of interest.


### Problem statement
Music genres are established conventions, but probably they might be distiguish-able via some music features. Indeed, as genres are many, and very often they are overlapping, it is expected to encounter some challenges to have high prediction performances. The project has been also conducted only exclusively on two better differentiated genres (pop and classical), and the performance as expected were much higher.
The problem in a multi-class classification problem. The aim is to explore the performance of the faesible classification algoritms and evaluate the performances on 10 different music genre. Classifiers that are suitable for multi-class classification will be used.

### Project overview
The assignment is divided into steps:

- data exploration and preparation
- I algorithm: Support Vector Classifier
- II algorithm: Nearest Neighbor
- III algorithm: Decision Tree
- IV algorithm: Ensamble Methods (Random Forest / Extra Trees plus Boosting)
- Analysis results and findings
- Extraction of music features from selected music
- Prediction of extra music

## Data Exploration and Prparation
Dataset consists in 10,000 entries with 100 entries per class/label. The classes are balanced and equally represented: jazz, blues, disco, classical, pop, reggae, metal, hiphop, country, rock.
The dataset has 30 columns of which: 1 is the label column, and 28 the the "features" columns.
The features are all continuous variables, extracted with the python library "LIBROSA". The "labels" ara categorical.
These are:


The features analysis suggests that they are almost normally distributed, which suggests the utilization of the StandardScaler() at a later stage, before launching the training algoritms. There are no categorical features to be encoded.
Performing a correlation analysis, it seems that some of the features are highly positively or negatively correlated.
These are the highly correlated features:



It is decided then to drop 'tempo', 'spectral centroid', 'mfcc2' and 'mfcc8' features, to reduce the correlation between features therefore not to undermine the performance of algoritms sensitive to correlation (such as RandomForest). 
The data explorative part shows how the problem is not linearly separable, this means that we should use non-linear classifiers.
To sum up before the training and test over the algorithms:
- the dataset is balanced: all classes are represented in the same proportion
- the features are almost all normally distributed, therefore StandardScaler will be used most of the times
- the 10 classes seems not to be linearly separable, making this a non-linear Multi-class Classification problem

## Algoritms

### Support Vector Classifier
Pipeline:
- Train / test split: stratified (to maintain classes proportions). 65% train size / 35% test size (as the dataset is relatively small, we keep a bigger chunk of data for the test
- Scaling using StandardScaler()
- Test of SVC different kernels with default parameters ('rbf', 'poly', 'sigmoid', 'linear' kernels) with cross_val_score()
- Parameters tuning with GridSearcCV()
- Launching train, fit and predict with best found parameters: SVC(kernel = 'rbf', C = 2.66, gamma = 0.05)
- Evaluating with "accuracy" parameter: 0.6542
- Plot confusion matrix

### KNearestNeighbors
Pipeline:
- Train / test split: stratified (to maintain classes proportions). 65% train size / 35% test size (as the dataset is relatively small, we keep a bigger chunk of data for the test
- Scaling using MinMaxScaler() (knn works slightly better with normalized (0,1) data
- Test of knn with default parameters (n_neighbors = 5) with cross_val_score()
- Parameters exploration
- Launching train, fit and predict with best found parameters: KNeighborsClassifier(n_neighbors = 8)
- Evaluating with "accuracy" parameter: 0.5542
- Plot confusion matrix

### Decision Tree
Pipeline:
- Train / test split: stratified (to maintain classes proportions). 65% train size / 35% test size (as the dataset is relatively small, we keep a bigger chunk of data for the test
- No scaling needed for trees classifiers
- Test of DecisionTreeClassifier() with default parameters with cross_val_score()
- Parameters exploration with GridSearchCV()
- Launching train, fit and predict with best found parameters: DecisionTreeClassifier(criterion = "entropy", min_samples_leaf = 3, random_state = 5, max_depth = 15)
- Evaluating with "accuracy" parameter: 0.4257
- Plot confusion matrix

### Ensamble: RandomForest and ExtraTrees with AdaBoost
Pipeline:
- Train / test split: stratified (to maintain classes proportions). 65% train size / 35% test size (as the dataset is relatively small, we keep a bigger chunk of data for the test
- No scaling needed for trees classifiers
- Test of RandomForestClassifier() and ExtraTreesClassifier() with default parameters with cross_val_score()
- Parameters exploration with GridSearchCV()
- Launching train, fit and predict with best found parameters: RandomForestClassifier(n_estimators = 150, max_features = 8) and ExtraTreesClassifier(n_estimators = 150, max_features = 8)
- Evaluating with "accuracy" parameter: RandomForest 0.6542, ExtraTrees 0.6885 
- Launching train, fit and predict AdaBoost on the better performing algoritm: AdaBoostClassifier(ExtraTreesClassifier(max_features = 8, n_estimators = 150), algorithm = "SAMME.R", learning_rate = 0.5)
- Evaluating with "accuracy" parameter: 0.7114
- Plot confusion matrix

## Analysis Results
Confronting the accuracy of the different algorithms:
- AdaBoost(ExtraTrees) 0.7114
- ExtraTrees 0.6885
- SVC 0.6542
- RandomForest 0.6542
- kNN 0.5542
- DecisionTree 0.4257

Some genres were more often mispredicted:
- Rock was often mispredicted for country and disco
- Raggae mispredicted for country and hiphop
- Jazz mispredicted for classical
- Disco mispredicted for hiphop

Some genres where more often predicted correctly:
- Classical and metal

Rock was most often misspredicted than other genres.

It seems clear that there are some genres that seems to be more distinguishable than others (classic and metal), instead, other genres are more overlapping and probably including more "blurred" features.

## Prediction over extra imported music
Data was imported utilising the same parameters as the original dataset, and data was preprocessed in the same way.
The best performing algorithms were used with the best identified parameters (SVC and AdaBoost with ExtraTrees) and trained in the same train/test split.
The performance of the two algorithms is quite different, making different predictions on the extra data.
Comparing the two outcomes, in 60% of the cases the same genre was predicted by the two algoritms, which is more or less in line with the performance of the algoritms explored.

The music submitted for prediction belongs to the world of independent music, therefore the genres predicted might be quite biased due to the different quality of audio production and processing. 

## References
