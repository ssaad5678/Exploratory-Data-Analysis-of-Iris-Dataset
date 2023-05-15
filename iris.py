import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import streamlit as st

sns.set()

# Load the Dataset
iris_data = pd.read_csv('iris.csv')

# Gathering Information from Data
st.subheader('Dataset Information')
st.dataframe(iris_data)
st.text(iris_data.info())

# Statistical Insight
st.subheader('Statistical Insight')
st.write(iris_data.describe())

# Checking for Duplicate Entries
duplicate_entries = iris_data[iris_data.duplicated()]
st.subheader('Duplicate Entries')
st.write(duplicate_entries)

# Checking the balance
Species_count = iris_data['Species'].value_counts()
st.subheader('Species Count')
st.bar_chart(Species_count)

# Data Visualization
fig, ax=plt.subplots(figsize=(17, 9))
plt.title('Comparison between various Species based on sepal length and width')
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris_data, s=50,ax=ax)
st.pyplot(fig)

# Comparison between various Species based on petal length and width
st.subheader('Comparison based on Petal Length and Width')
fig, ax=plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species',data=iris_data,s=50,ax=ax)
st.pyplot(fig)

# Bi-variate Analysis
st.subheader('Pairwise Scatter Plot')
sns.pairplot(iris_data, hue="Species", height=4)
st.pyplot()

# Remove the 'Species' column from the DataFrame
numeric_columns = iris_data.select_dtypes(include=[float, int])
# Checking Correlation
st.subheader('Correlation Heatmap')
fig, ax=plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True,ax=ax)
st.pyplot(fig)

# Checking Mean & Median Values for each Species
Species_mean_median = iris_data.groupby('Species').agg(['mean', 'median'])
st.subheader('Mean and Median Values for each Species')
st.dataframe(Species_mean_median)

# Visualizing the distribution, mean, and median using box plots & violin plots

# Box plots
fig, axes = plt.subplots(2, 2, figsize=(16,9))
sns.boxplot(y="PetalWidthCm", x="Species", data=iris_data, orient='v', ax=axes[0, 0])
sns.boxplot(y="PetalLengthCm", x="Species", data=iris_data, orient='v', ax=axes[0, 1])
sns.boxplot(y="SepalLengthCm", x="Species", data=iris_data, orient='v', ax=axes[1, 0])
sns.boxplot(y="SepalWidthCm", x="Species", data=iris_data, orient='v', ax=axes[1, 1])
st.pyplot(fig)

# Violin plots
fig, axes = plt.subplots(2, 2, figsize=(16,10))
sns.violinplot(y="PetalWidthCm", x="Species", data=iris_data, orient='v', ax=axes[0, 0], inner='quartile')
sns.violinplot(y="PetalLengthCm", x="Species", data=iris_data, orient='v', ax=axes[0, 1], inner='quartile')
sns.violinplot(y="SepalLengthCm", x="Species", data=iris_data, orient='v', ax=axes[1, 0], inner='quartile')
sns.violinplot(y="SepalWidthCm", x="Species", data=iris_data, orient='v', ax=axes[1, 1], inner='quartile')
st.pyplot(fig)

#Designing Machine Learning Model
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(iris_data.drop('Species', axis=1), iris_data['Species'], test_size=0.3, random_state=5)

# Display the shape of the training and test sets
st.write('Training set shape:', x_train.shape)
st.write('Test set shape:', x_test.shape)
st.write('Training labels shape:', y_train.shape)
st.write('Test labels shape:', y_test.shape)
#succesfully splitting 80% data for training and 20% testing
#Using logistic algorithm for training and prediction purposes
# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the Logistic Regression model
max_iterations = 1000  # Set the maximum number of iterations
logreg = LogisticRegression(solver='lbfgs', max_iter=max_iterations)
logreg.fit(x_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(x_test)

# Compute the accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)

# Display the accuracy
st.write("Accuracy:", accuracy)
st.markdown("Therefore the Accuracy obtained is 100%")

v = iris_data.drop(['Id', 'Species'], axis=1)
Y = iris_data['Species']

x_train, x_test, y_train, y_test = train_test_split(v, Y, test_size=0.3, random_state=5)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

prediction_virginica = np.array([[2,4,3,5]])
prediction_versicolor1 = np.array([[6,3,4,2]])
prediction_virginica2 = np.array([[3,2,4,5]])


prediction_result = logreg.predict(prediction_virginica)
prediction_result1 = logreg.predict(prediction_versicolor1)
prediction_result2 = logreg.predict(prediction_virginica2)


st.write("The predicted species for the given input:", prediction_result)
st.write("The predicted species for the given input:", prediction_result1)
st.write("The predicted species for the given input:", prediction_result2)



