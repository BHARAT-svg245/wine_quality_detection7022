# wine_quality_detection7022
Wine_Quality_detection App
it is machine learning web apps making with streamlit to predict the quality of wine.
#Description about the data
The data has 11 features with 1000 samples.
Data has wine quality class-(3,4,5,6,7,8),but the samples of class-(3,4,8) is very small,and this affects accuracy of our model.So,i remove these labels sample to get the better accuracy.
We can oversample these small labels class,so that accuracy can increase. 
All features have their own role in training the machine learning models.
#Description about the app
This app uses the of .csv file attached with it and trained itself.
Then it saves the weights,biases and other features of the data required for prediction.
I tried different machine learning algorithms to trained it,then getting the accuracy from them.but RandomforestClassifier does it best with accuracy of 68%.
I make a small streamlit web app to enter the inputs by the user and  predict the quality of wine.
it is a very basic web apps.
##Requirements
1.python
2.Library of python-Numpy,Pandas,Scikit-learn,joblib
3.A dataset with a number of data samples to train the model.
4.Matplotlib and Seaborn library for exploratory data analysis.
#Results
After trying various machine learning classification techniques ,i got 68% accuracy

This line is for practice purpose of git and github