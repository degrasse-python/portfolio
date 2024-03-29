# Wine Classification
 
This repo is one part of a larger portfolio of projects I have completed. If you would like to see my other research work in weather and climate data feel free to contact me. All projects here are authored by Allen Deon Saunders. The '.dat' files in the folder are the source data files. The '.csv' and 'npy' files were produced by the author of the model in this repo for input into the classification model. 

## Project Description 

This model will classify the rank and type of wine by quality using the data generated by the [source](https://github.com/bucoyas/ml_portfolio/tree/master/wine_quality/data). 
The description of the data is provided by the source. 

### Introduction
You want your wine to age like wine - not vinegar. Understanding what makes a wine top shelf is beneficial to not only vineyards - this information is also valuable to the average person. This report shows the relationships between wine characteristics and quality for red and white variants of the Portuguese "Vinho Verde" wine. For vineyards who produce this wine variety a single report can change their perspective on how their process of winemaking affects wine quality. To have a decent understanding of the characteristics of which properties contribute to quality of wine is to be a head above the rest. The process of making wine differs from company to company but even the best creators are always looking for new insights. The public should have an understanding of what makes our wine delicious. 

Using data about the wine from two datasets of red and white variants of the Portuguese "Vinho Verde" wine we will explore how characterics like volatile acid, citric acid, chloride concentration, total sulfates dissolved, pH, and more effect the quality of wine. Potential clients can see how these factors affect the quality of wine. These insights will give a winemaker clarity about how their wine may potentially rank against others of the same wine variety.  

#### Datasets
Two datasets were created, using red and white wine samples. The inputs include objective tests (e.g. PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). Missing Attribute Values: None. The details are described in [Cortez et al., 2009]. 
   Variables:
   1 - fixed acidity, 2 - volatile acidity, 3 - citric acid, 4 - residual sugar, 5 - chloride, 6 - free sulfur dioxide, 7 - total sulfur dioxide, 8 - density, 9 - pH, 10 - sulphates, 11 - alcohol, 12 - quality (score between 0 and 10)
#### Data importation and cleaning
The data was imported using PANDAS the python 3rd party open source library used for data analysis. The data was clean and came out of the box with separated values with a semicolon “;” as a separator. No preprocessing was performed. A check to see if there were missing values and there are none. 

### DATA EXPLORATION
#### Relations

The density was examined with respect to wine quality inspected. We immediately notice the distribution for quality seems normal. Yet, we look for normality in the characterics of the wine with respect to quality. Here it would be best to have a uniform distribution of quality to be able to compare the wine by quality. Because there are a low number of instances of bad and great wine we will pool those together once we start the hypothesis. The groups are low[3,4], mid[5,6], high[>6]. This grouping should eliminate noise from having a small samples of 8,9,3,4 ranking wines. There was also a high variance of among the features with respect to each quality. Normalized boxplots were used to show a side by side comparison of attributes without the scales skewing the message shown by the variance of each feature. Outliers were removed using a interquartile range method before performing any frequentist or Bayesian statistical tactics. To explore the relationships between attributes I used a matrix heatmap for robust visualization. 

From the matrix plots we can see strong relationships between various combinations of wine attributes for both white and red wine, respectively. The correlations are explored in the regression plots later in the report. 

Building a model requires a detailed level of data exploration to show  contrast between positive and negative relationship between the target and the attributes. This level of data exploration is required before any decent model is chosen. For this kind of problem Simple regression is not adequate to predict wine quality based on these features. A simple and dry solution would be to build a fixed effects or random effects model. Anything more simple than these could not do the job. After examining the strongest relationships between attributes I selected pairs or attributes with a threshold using absolute r value of greater than or equal to 0.1.

### Feature engineering introduction 
The variability was quite large for most of the features so removal of outliers was performed using an interquartile range method before computing any statistical tests. Outliers were removed using the Tukey’s IQR method. From the results of the one way ANOVA we can see that with the exception of residual sugar for red wine all of the attributes of each wine have means that are different. To explore this as a random effects model a comparison among treatment means was executed the Tukey’s comparison of means. Also it should be noted that for the white and red wine the higher quality and lower quality had such low numbers that they could not be included in the statistical testing with attributes which had much higher instances. As we can see from the summary all of the groups of wine are fundamental different in quality. And, there is an observed mean difference in the groups. For this reason the dataset was split by quality into groups: low, mid, and high quality. After the analysis of variance we can see the only feature in which we fail to reject the null hypothesis is `resid_sugar` from the Red Wine dataset. This means the variances of features are different. From the Tukey comparisons of means we can see the mean which is the same is `resid_sugar`. Other features have means which are equal among groups yet we can reject the null hypothesis for the other combinations. This leads us to believe that the problem scope is more difficult than previously believed. The features which we reject the null hypothesis for are `fixed_acidity`, `vol_acidity`, and `free_sul_dioxide` - for white wine. The features in which we reject the null hypothesis are `vol_acidity`, `citric_acid`, and `sulphates` - for red wine. These are going to be the most divisive features according to our classic statistical tests. 

Without a deep dive into topological data analysis we will do a greedy search to see which algorithms would work best on this problem. It is very clear we have labelled data which makes our search point toward supervised learning algorithms. The output from our model is a class which makes this a classification problem. Before we perform our exhaustive search we will look at the collinearity of the features using principal data analysis which will give us a visualization of the collinearity. From the white wine PCA plots below we can see the collinearity much more clearly than using regression based methods. We can see that `chlorides`, `density`,  and `sulphates` are highly correlated. The same is for `total_SD`, `citric_acid`, `free_sul_dioxide`. This leaves us to pick the best from the groups and feed them into a model. The vector represents the coefficient of the feature so it is best to remove the weakest of the correlated features. `total_SD`, `density`,  and `sulphates`. We choose to leave `free_sul_dioxide` because it is far enough from `citric_acid` that we can investigate later if it is harming our model or not. We can see that the classical VIF score of feature selection is primitive to PCA. The high collinearity between features tells us immediately that modeling with simple statistical predictions like logistic regression will not be sufficient enough for the complexity of the dataset. This means the algorithms that will find the best solution set will be non linear models. Yet we will attempt to use logistic regression and random forest decision trees to see if they compare. 

### Sampling Technique
A dataset is imbalanced if the classification categories are not approximately equally represented. Often real-world data sets are predominately composed of “normal” examples with only a small percentage of “abnormal” or “interesting” examples. It is also the case that the cost of misclassifying an abnormal (interesting) example as a normal example is often much higher than the cost of the reverse error. Under-sampling of the majority (normal) class has been proposed as a good means of increasing the sensitivity of a classifier to the minority class. 

#### Synthetic Minority Oversampling (SMOTE)
over-sampling approach in which the minority class is over-sampled by creating “synthetic” examples rather than by over-sampling with replacement. The minority class is over-sampled by taking each minority class sample and introducing synthetic examples along the line segments joining any/all of the k minority class nearest neighbors. Depending upon the amount of over-sampling required, neighbors from the k nearest neighbors are randomly chosen. The default arguement for SMOTE is 3 nearest neighbors unless passed to the function different. Synthetic samples are generated in the following way: Take the difference between the feature vector (sample) under consideration and its nearest neighbor. Multiply this difference by a random number
between 0 and 1, and add it to the feature vector under consideration. This causes the selection of a random point along the line segment between two specific features. This approach effectively forces the decision region of the minority class to become more general.

#### Adaptive Synthetic (ADASYN)
ADASYN is based on the idea of adaptively generating minority data samples according to their distributions: more synthetic data is generated for minority class samples that are harder to learn compared to those minority samples that are easier to learn. The ADASYN method can not only reduce the learning bias introduced by the original imbalance data distribution, but can
also adaptively shift the decision boundary to focus on those difficult to learn samples. The key idea of ADASYN algorithm is to use a density distribution rˆi as a criterion to automatically decide the number of synthetic samples that need to be generated for each minority data example. Physically, rˆi is a measurement of the distribution of weights for different minority class examples according to their level of difficulty in learning. The resulting dataset post ADASYN will not only provide a balanced representation of the data distribution (according to the desired balance level defined by the β coefficient), but it will also force the learning algorithm to focus on those difficult to learn examples. This is a major difference compared to the SMOTE algorithm, in which equal numbers of synthetic samples are generated for each minority data example.

#### Principal Component Analysis - sklearn
Without a deep dive into topological data analysis we will do a greedy search to see which algorithms would work best on this problem. It is very clear we have labelled data which makes our search point toward supervised learning algorithms. The output from our model is a class which makes this a classification problem. Before we perform our exhaustive search we will look at the collinearity of the features using principal data analysis which will give us a visualization of the collinearity. From the white wine PCA plots below we can see the collinearity much more clearly than using regression based methods. The goal of PCA analysis, with respect to feature engineering, is to get features with the highest degree of orthognality to each other. 

##### White wine PCA
We can see that `chlorides`, `density`,  and `sulphates` are highly correlated. The same is for `total_SD`, `citric_acid`, `free_sul_dioxide`. This leaves us to pick the best from the groups and feed them into a model. The vector represents the coefficient of the feature so it is best to remove the weakest of the correlated features. `total_SD`, `density`,  and `sulphates`. We choose to leave `free_sul_dioxide` because it is far enough from `citric_acid` that we can investigate later if it is harming our model or not. We can see that the classical VIF score of feature selection is primitive to PCA. The high collinearity between features tells us immediately that modeling with simple statistical predictions like logistic regression will not be sufficient enough for the complexity of the dataset. This means the algorithms that will find the best solution set will be non linear models. Yet we will attempt to use logistic regression and random forest decision trees to see if they compare. From the PCA plot after removal of these feature we observe an increase in orthognality.

##### Red wine PCA
For the red wine classification we observe residual sugar is the only feature which has equal variance and mean amongst all quality levels - via ANOVA. This tells us `resid_sugar` is the worse feature and will corrupt any model. From the Tukey's comparison of means we observe red wine's most divisive features are `sulphates`, `citric_acid`, and, `vol_acidity`.The others are questionable due to the them having equal means among some comparisons. From the PCA plot below we can see the colinearity of the features from the red wine dataset. We can see that `chlorides`, and `density`  are highly correlated so trash `density`. As well as `free_sul_dioxide` and `total_SD` so we trash `total_SD`. And the same is for `fixed_acidity`, `citric_acid`,  and `sulphates` so we trash `fixed_acidity`. The trash features are the features which had the lowest mean difference from the comparison of mean. From Tukey's comparision of means we know the most divisive features are `vol_acidity`, `citric_acid`, and `sulphates` which is why those were kept. We choose to leave ditch `resid_sugar` because it failed the ANOVA and all of the Tukey's comparison of means. From the PCA plot after removal of these feature we observe an increase in orthognality.

### Statistical Learning Model Evaluation

To select the correct model we must first evaluate the scope of the problem we are trying to solve. The data we have is numerical to predict our labeled categorical classes. This tells us we have a classification problem in which we will us supervised learning to predict wine quality. We will attempt to classify the original labels and by using a grouping method by grouping the wine classes into high, mid, and low quality. The decision was made to perform this grouping due to the low count at the low and high ends of the dataset as well as the subjective collection method of the dataset. Because of the imbalanced form of the data we will use the imblearn library and use classifiers with boosting. Apart from the random sampling with replacement, there are two popular methods to over-sample minority classes: (i) the Synthetic Minority Oversampling Technique (SMOTE) [CBHK2002] and (ii) the Adaptive Synthetic (ADASYN) [HBGL2008] sampling method. These algorithms can be used in the same manner. We will evaluate each sampling method with grid searching of parameters to generate a solution to the classification problem. In addition to the imbalanced data his data has features with high collinearity and high dimensionality which means any linear model will not be sufficient enough make correct predictions. We will grid search over a set of parameters to find the optimal settings for each model as well as check to see if normalizing or PCA is the transformation for the data before feeding it into each model.


#### Logistic Regression.
##### White Wine
Classification with original labels 
Logistic Regression was not fit to capture the complexity of the dataset. Using standard scalar transformation and SMOTE sampling method from the imblearn library I was able to boost the balanced accuracy from 19 to 42 percent. 

Classification with grouped labels
Grouping the labels into low, mid, high improved the classification. Without sampling logistic regression had balanced accuracy of %36 using the same model parameters. Using ANASYN sampling we were able to reach %62 with the sample model parameters.

##### Red Wine
Classification with original labels 
Logistic Regression was not fit to capture the complexity of the dataset. Using standard scalar transformation and SMOTE sampling method from the imblearn library I was able to boost the balanced accuracy from 29 to 56 percent. This was attained using a C of 10 and random state of 0.

Classification with grouped labels
Grouping the labels into low, mid, high improved the classification. Without sampling logistic regression with standard scalar transformation of the dataset had balanced accuracy of %44 using a C of ~0.4641 and random state of 0. Using ANASYN sampling we were able to reach %71 with the sample model parameters.


#### k Nearest Neighbors.
##### White Wine
Classification with original labels 
K Nearest Neighbors was the best classifier of the original labels with and without using the sampling techniques. Without sampling techniques the best classifier using one nearest neighbor and uniform distancing with a balanced accuracy across labels of %38. Using either SMOTE or ANASYN sampling techniques, standard scalar transformation, one nearest neighbors and uniform weighting on distanced points the classifier had an increased balanced accuracy of %82.

Classification with grouped labels
Grouping the labels improved the classification. kNN had the 2nd best classification of grouped labels balanced accuracy of %62 without sampling using 3 nearest neighbor, and weighting each point using an inverse distance. Using SMOTE sampling technique of the grouped labels, standard scalar transformation of the features, 4 nearest neighbors and uniform weighting on distanced points the classifier had an increased balanced accuracy of %83.



##### Red Wine
Classification with original labels 
K Nearest Neighbors was the best classifier of the original labels with and without using the sampling techniques. Without sampling techniques the best classifier using standard scaling, 4 nearest neighbor and inverse distance weighting of points with a balanced accuracy across labels of %32. Using SMOTE sampling techniques, standard scalar transformation, one nearest neighbor and uniform weighting on distanced points the classifier had an increased balanced accuracy of %81.

Classification with grouped labels
Grouping the labels improved the classification. kNN had the 2nd best classification of grouped labels balanced accuracy of %56 without sampling using 2 nearest neighbor, and weighting each point uniform. Using either SMOTE or ANASYN sampling technique of the grouped labels, standard scalar transformation of the features, 1 nearest neighbors and uniform weighting on distanced points the classifier had an increased balanced accuracy of %90.


#### Random Forest Regression.
##### White Wine
Classification with original labels 
Even with the popularity of Random Forest algorithm this model was not expected to perform with highly dimensional data. Ranking 3rd in the classification of the original labels the balanced accuracy is %61 with parameters: max depth of 6, minimum samples for leafs of 2, number of estimators was 100, random state equal to 2, standard scalar transform of input data, and SMOTE sample. This was an increase from %22 without sampling the imbalanced dataset using the same parameters. 

Classification with grouped labels
Ranking 3rd in the classification of the original labels without resampling the dataset the balanced accuracy is %39 with parameters: max depth of 6, minimum samples for leafs of 2, number of estimators was 100, random state equal to 2, and standard scalar transform of input data. Using the This was an increase from %22 without sampling the imbalanced dataset using the same parameters. 




##### Red Wine
Classification with original labels 
Even with the popularity of Random Forest algorithm this model was not expected to perform with highly dimensional data. Ranking 3rd in the classification of the original labels the balanced accuracy is %61 with parameters: max depth of 6, max features of 6, minimum samples for leafs of 3, number of estimators was 200, random state equal to 2, standard scalar transform of input data, and SMOTE sample. This was an increase from %29 without sampling the imbalanced dataset using the same parameters. 


Classification with grouped labels
Grouping of the labels increased classification balanced accuracy. Without sampling techniques and parameters of max depth of 6, max features of 5, min samples for leaf of 3, number of estimators of 200, random state of 2 the random forest classifier had a balanced accuracy of 83. 



#### Support Vector Machines.
##### White Wine
Classification with original labels 
Not surprised at the performance of this algorithm against this highly dimensional dataset. For classification of original labels without sampling techniques this was the 3rd best classifier with a balanced accuracy of %36 using parameters of: radial basis kernel, classification based on probability estimates (softmax), balanced class weight, C of 15, Kernel gamma coefficient of 1. Sampling greatly increased the classification of this algorithm to have a %81 using SMOTE sampling with the same model parameters.  


Classification with grouped labels
This was the best classifier using the SVMSMOTE sampling technique of the grouped labels with a balanced accuracy of %88 and parameters: radial basis kernel, classification based on probability estimates (softmax), balanced class weight, C of 15, gamma of 10. Without sampling, a radial basis kernel, classification based on probability estimates (softmax), balanced class weight, C of 1, gamma of 1 the balanced accuracy %57.


##### Red Wine
Classification with original labels 
For classification of original labels without sampling techniques this was the 2nd best classifier with a balanced accuracy of %30 using parameters of: radial basis kernel, classification based on probability estimates (softmax), balanced class weight, C of 15, Kernel gamma coefficient of 1. Sampling greatly increased the classification of this algorithm to have a %81 using SMOTE sampling with the same model parameters.  


Classification with grouped labels
This was the best classifier using sampling techniques of the grouped labels with a balanced accuracy of %94. This was achieved using principal component analysis of the features with 6 components and parameters: radial basis kernel, classification based on probability estimates (softmax), balanced class weight, C of 15, gamma of 10. Without sampling, a radial basis kernel, classification based on probability estimates (softmax), balanced class weight, C of 1, gamma of 1 the balanced accuracy %57.

### Best Results from model evaluation


|                           White Wine Model Selection Analysis                                               |   Balanced Accuracy | Sampling   | Transformation   |
|:--------------------------------------------------------------------------------|--------------------:|:-----------|:-----------------|
| ('Grouped', "<class 'sklearn.ensemble.forest.RandomForestClassifier'>")         |            0.821039 | no         | scaler           |
| ('Grouped', "<class 'sklearn.linear_model.logistic.LogisticRegression'>")       |            0.718517 | no         | scaler           |
| ('Grouped', "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>")  |            0.909497 | no         | Scale            |
| ('Grouped', "<class 'sklearn.svm.classes.SVC'>")                                |            0.929241 | no         | scaler           |
| ('Original', "<class 'sklearn.ensemble.forest.RandomForestClassifier'>")        |            0.719906 | no         | scaler           |
| ('Original', "<class 'sklearn.linear_model.logistic.LogisticRegression'>")      |            0.574958 | no         | scaler           |
| ('Original', "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>") |            0.823676 | no         | Scale            |
| ('Original', "<class 'sklearn.svm.classes.SVC'>")                               |            0.825491 | no         | scaler           |



|      Red Wine Model Selection Analysis                                                                  |   Balanced Accuracy | Sampling   | Transformation   |
|:--------------------------------------------------------------------------------|--------------------:|:-----------|:-----------------|
| ('Grouped', "<class 'sklearn.ensemble.forest.RandomForestClassifier'>")         |            0.821039 | no         | scaler           |
| ('Grouped', "<class 'sklearn.linear_model.logistic.LogisticRegression'>")       |            0.718517 | no         | scaler           |
| ('Grouped', "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>")  |            0.909497 | no         | Scale            |
| ('Grouped', "<class 'sklearn.svm.classes.SVC'>")                                |            0.929241 | no         | scaler           |
| ('Original', "<class 'sklearn.ensemble.forest.RandomForestClassifier'>")        |            0.719906 | no         | scaler           |
| ('Original', "<class 'sklearn.linear_model.logistic.LogisticRegression'>")      |            0.574958 | no         | scaler           |
| ('Original', "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>") |            0.823676 | no         | Scale            |
| ('Original', "<class 'sklearn.svm.classes.SVC'>")                               |            0.825491 | no         | scaler           |
