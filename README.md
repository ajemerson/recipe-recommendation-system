##  [CSC722: Advanced Topics in Machine Learning Project] Recipe Recommendation System
# Project Goal: 
Data-driven recipe recommendation system using web-scraped recipe data (including but not limited to data like ingredients, health facts, etc.) and user’s historical preference.
# Motivation:
Cooking is a hobby for some and a major problem for others. However, you can always use a helping hand for cooking. Being a student, it is always a difficult decision to decide what to eat for lunch or dinner. Sometimes faced with limited items in the kitchen, it is always a challenge to decide what to cook for meal. This inspired us to create a system that can recommend recipes based on ingredient suggestions.
# Data:
All data is stored on a Google Drive at (TODO: provide access to data).
We scraped a total of **61,900** recipes, and parsed the set of ingredients to produce a set of unique ingredients. We then passed these unique ingredients through a unit conversion pipeline, resulting in a total of 37,765 ingredient features (each ingredient has a 'mass' feature and a 'volume' feature). Due to the dimensionality issues, we have processed the data several times using feature selection, and we have uploaded the smaller representations.
- **recipe_info.csv**: the recipe 'name', 'type' (e.g., Breakfast and Brunch), and 'subtype' (e.g., Pancake Recipes)
- **recipes_normalized_varFS.csv**: the set of ingredients, normalized, for all recipes. There are a total of 7,580 features in this processed set after performing Variance Threshold feature selection, using the mean normalized variance for all features.
- **recipes_normalized_varFS_incrementalPCA100.csv**: set of ingredients transformed to a size of (61900, 100) by using incremental PCA in batches of 100.
- **recipes_normalized_varFS_extraTrees912.csv**: set of ingredients chosen by ensemble classifier (classifying subtype) with a size of (61900, 912)
- **recipes_normalized_varFS_extraTrees186.csv**: set of ingredients chosen again (from previously derived set) by an identical ensemble classifier, with a size of (61900, 186)
- **recipes_encoded100.csv**: recipe dimensions were reduced from 7000+ to 100 using a stacked autoencoder of hidden layers 1000 and 100. Input Set: recipes_normalized_varFS.csv. Output Size: (61900, 100)
- **recipes_encoded1000.csv**: recipe dimensions were reduced from 7000+ to 1000 using a one-hidden-layer autoencoder. Input Set: recipes_normalized_varFS.csv. Output Size: (61900, 1000)
# References:
- A possible basis for our own dataset: https://www.kaggle.com/hugodarwood/epirecipes 
- Resource for dataset (includes tags): https://www.epicurious.com/search/ 
- Resource for dataset (can search by meal type): https://www.allrecipes.com/recipes/
- GAN paper: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
- GAN blog: https://towardsdatascience.com/generative-adversarial-networks-explained-34472718707a
- GMM slide deck: http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf
- Conditional GAN: https://arxiv.org/abs/1411.1784
- Recommendation Systems Overview: http://infolab.stanford.edu/~ullman/mmds/ch9.pdf 
