##  [CSC722: Advanced Topics in Machine Learning Project] Recipe Recommendation System
# Requirements:
- **Python 3.7**: This project assumes that dictionaries have some notion of consistent ordering. By default, 3.7 makes this assumption. Otherwise, dictionaries should be changed to OrderedDict.
- **TO DEMO:**
>> 1. Download recipe_info.csv, recipes_encoded100.csv, and recipes_normalized_varFS_extraTrees186.csv from the Google Drive link below
>> 2. Run recipe_rate_gui.py and select the type of data-cluster combination desired. Bon Appetit!
# Project Goal: 
Data-driven recipe recommendation system using web-scraped recipe data (including but not limited to data like ingredients, health facts, etc.) and user’s historical preference.
# Motivation:
Cooking is a hobby for some and a major problem for others. However, you can always use a helping hand for cooking. Being a student, it is always a difficult decision to decide what to eat for lunch or dinner. Sometimes faced with limited items in the kitchen, it is always a challenge to decide what to cook for meal. This inspired us to create a system that can recommend recipes based on ingredient suggestions.
# Data:
All data is stored on a Google Drive at https://tinyurl.com/yajzjchs.
We scraped a total of **61,880** recipes, and parsed the set of ingredients to produce a set of unique ingredients. We then passed these unique ingredients through a unit conversion pipeline, resulting in a total of 37,765 ingredient features (each ingredient has a 'mass' feature and a 'volume' feature). Due to the dimensionality issues, we have processed the data several times using feature selection, and we have uploaded the smaller representations.
- **recipe_info.csv**: the recipe 'name', 'type' (e.g., Breakfast and Brunch), and 'subtype' (e.g., Pancake Recipes). Also contains a column for each GMM-produced cluster labels. For each of the feature-reduced datasets, there are cluster results for 4, 25, and 54 components.
- **recipes_normalized_varFS.csv**: the set of ingredients, normalized, for all recipes. There are a total of 7,580 features in this processed set after performing Variance Threshold feature selection, using the mean normalized variance for all features.
- **recipes_normalized_varFS_incrementalPCA100.csv**: set of ingredients transformed to a size of (61880, 100) by using incremental PCA in batches of 100.
- **recipes_normalized_varFS_incrementalPCA1000.csv**: set of ingredients transformed to a size of (61880, 1000) by using incremental PCA in batches of 1000.
- **recipes_normalized_varFS_extraTrees912.csv**: set of ingredients chosen by ensemble classifier (classifying subtype) with a size of (61880, 912)
- **recipes_normalized_varFS_extraTrees186.csv**: set of ingredients chosen again (from previously derived set) by an identical ensemble classifier, with a size of (61880, 186)
- **recipes_encoded100.csv**: recipe dimensions were reduced from 7000+ to 100 using a stacked autoencoder of hidden layers 1000 and 100. Input Set: recipes_normalized_varFS.csv. Output Size: (61880, 100)
- **recipes_encoded1000.csv**: recipe dimensions were reduced from 7000+ to 1000 using a one-hidden-layer autoencoder. Input Set: recipes_normalized_varFS.csv. Output Size: (61880, 1000)
# References:
- https://scikitlearn.org/stable/modules/generated/sklearn.metrics.calinski_harabaz_score.html#sklearn.metrics.calinski_harabaz_score
- https://scikit-learn.org/stable/modules/mixture.html
- http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf
- http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. nature, 521(7553), 436.
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
- https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- https://www.kaggle.com/kaggle/recipe-ingredients-dataset
- https://www.allrecipes.com
- Rätsch, G., Onoda, T., & Müller, K. R. (2001). Soft margins for AdaBoost. Machine learning, 42(3), 287-320.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
Balabanović, M., & Shoham, Y. (1997). Fab: content-based, collaborative recommendation. Communications of the ACM, 40(3), 66-72.
