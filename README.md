# Goal
In this series of practices, we will go through the theory and concepts of different types of classification along with the applications by using Python.

## Algorithm type
Below are the different types of the machine learning algorithms:

#### Prediction
  - Supervised
    - Continuous variables: Regression
      - Gradient Descent
      - Gradient Ascent
    - Discrete variables: Classification
      - kNN
      - Decision Tree
      - Random Forest
      - Naive Bayes
      - Expectation Maximization
      - Support Vector Machines

# Classification
We will go through the theory and applications of different types of classification.

## k-Nearest Neighbors (kNN)
How kNN works:
1. As in the general problem of classification, we have a set of data points for which we know the correct class labels.
2. When we get a new data point, we compare it to each of our existing data points and find similarity.
3. Take the most similar k data points (k nearest neighbors).
4. From these k data points, take the majority vote of their labels. The winning label is the label/class of the new data point.

Now we can start kNN with the wheat dataset. The dataset comprises of data about kernels belonging to three different varieties of wheat: Kama, Rosa and Canadian. 
These are represented as 1, 2, and 3 in the last column. For each wheat variety, with a random sample of 70 elements, high quality visualization of the internal kernel structure was detected using a soft X-ray technique. Seven geometric parameters of wheat kernels were measured, and use these measures to classify the wheat variety.

We can use a sizable portion (70-80%) to train a kNN classifier with different values of k and use the remaining data for testing.
### 0. Set up
```ruby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('wheat.txt', header=None, delimiter='\t+', engine="python")
```
Split the dataset into training and testing set
```
X_train, X_test, y_train, y_test = train_test_split(df[[0,1,2,3,4,5,6]],df[7], test_size=0.3)
```
### 1. Data training
Define the classifier using kNN function and train it
```
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)
```
Test the classifier by giving it test instances
```
prediction = classifier.predict(X_test)
```
Count how many were correctly classified
```
correct = np.where(prediction==y_test, 1, 0).sum()
print(correct)
```
--> 55
```
accuracy = correct/len(y_test)
print(accuracy)
```
--> 0.873015873015873

Run kNN for loop to see the accuracy with different amount of k
```ruby
results = []

for k in range(1, 51, 2):
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(X_train,y_train)
  prediction = classifier.predict(X_test)
  correct = np.where(prediction==y_test, 1, 0).sum()
  accuracy = correct/len(y_test)
  print ("k=", k, " Accuracy=", accuracy)
  results.append([k,accuracy])

# Convert that series of tuples in a dataframe for easy plotting
results = pd.DataFrame(results, columns=["k","accuracy"])
```
```
# Result
k= 1  Accuracy= 0.873015873015873
k= 3  Accuracy= 0.873015873015873
k= 5  Accuracy= 0.8571428571428571
k= 7  Accuracy= 0.8888888888888888
k= 9  Accuracy= 0.9206349206349206
k= 11  Accuracy= 0.9206349206349206
k= 13  Accuracy= 0.9365079365079365
k= 15  Accuracy= 0.9206349206349206
k= 17  Accuracy= 0.9206349206349206
k= 19  Accuracy= 0.9206349206349206
k= 21  Accuracy= 0.9206349206349206
k= 23  Accuracy= 0.873015873015873
k= 25  Accuracy= 0.873015873015873
k= 27  Accuracy= 0.873015873015873
k= 29  Accuracy= 0.873015873015873
k= 31  Accuracy= 0.873015873015873
k= 33  Accuracy= 0.873015873015873
k= 35  Accuracy= 0.8571428571428571
k= 37  Accuracy= 0.8571428571428571
k= 39  Accuracy= 0.8571428571428571
k= 41  Accuracy= 0.8571428571428571
k= 43  Accuracy= 0.8412698412698413
k= 45  Accuracy= 0.8412698412698413
k= 47  Accuracy= 0.8571428571428571
k= 49  Accuracy= 0.873015873015873
```

### 2. Visualization
```ruby
plt.plot(results.k, results.accuracy)
plt.title("Value of k and corresponding classification accuracy")
plt.show()
```
<img width="566" alt="Screenshot 2024-02-13 at 8 24 55 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/2b88eec2-5b1a-4fab-a60a-c1b81a35f68e">

## Decision Tree
The goal of the decision tree is to create a model that predicts the value of a target variable based on several input variables. A decision tree builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. 

We can start doing decision tree with the dataset which is sourced from the study “The Effects of a Joke on Tipping When it is Delivered at the Same Time as the Bill,” by Nicholas Gueguen (2002).

Can telling a joke affect whether or not a waiter in a coffee bar receives a tip from a customer? This study investigated this question at a coffee bar at a famous resort on the west coast of France. The waiter randomly assigned coffee-ordering customers to one of three groups: When receiving the bill one group also received a card telling a joke, another group received a card containing an advertisement for a local restaurant, and a third group received no card at all. He recorded whether or not each customer left a tip.

#### Variables
```
Card  Type of card used: Ad, Joke, or None
Tip   1=customer left a tip or 0=no tip
Ad    Indicator for Ad card
Joke  Indicator for Joke card
None  Indicator for no card
```
### 0. Set up
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv('TipJoke.csv')
```
### 1. Variable setting
Define the variables and the outcome you want to calculate
```
X = df[['Ad','Joke','None']]
y = df['Tip']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
```
### 2. Decision Tree Classifier
```ruby
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```
0.6875
[[44  0]
 [20  0]]

### 3. Visualization
```ruby
import graphviz
dot_data = tree.export_graphviz(dtree,out_file=None,feature_names=('Ad','Joke','None'),
                                class_names=('0','1'),
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render('tip_joke',view=True)

with open("tip_joke") as f:
  dot_graph = f.read()
graphviz.Source(dot_graph)
```
<img width="483" alt="Screenshot 2024-02-13 at 8 41 56 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/50863b54-45b0-4b4f-a7f5-fda952624dd7">

## Random Forest
One big problem the decision tree algorithm has is that it could overfit the data. What does that mean? It means it could try to model the given data so well that while the classification accuracy on that dataset would be wonderful, the model may find itself crippled when looking at any new data; it learned too much from the data.

One way to address this problem is to use not just one, not just two, but many decision trees, each one created slightly differently. And then take some kind of average from what these trees decide and predict. Such an approach is so useful and desirable in many situations where there is a whole set of algorithms that apply them. They are called ensemble methods.

In machine learning, ensemble methods rely on multiple learning algorithms to obtain better prediction accuracy than what any of the constituent learning algorithms can achieve. In general, an ensemble algorithm consists of a concrete and finite set of alternate models but incorporates a much more flexible structure among those alternatives. One example of an ensemble method is random forest, which can be used for both regression and classification tasks.

We can start doing random forest with the Blues Guitarists Hand Posture and Thumbing Style by Region and Birth Period data. This dataset has 93 entries of various blues guitarists born between 1874 and 1940. 

#### Variables
```
Regions:        1 means East, 2 means Delta, 3 means Texas.
Years:          0 for those born before 1906, 1 for the rest
Hand postures:  1= Extended, 2= Stacked, 3=Lutiform
Thumb styles:   Between 1 and 3, 1=Alternating, 2=Utility, 3=Dead
```

Using decision tree on this dataset, how accurately you can tell their birth year from their hand postures and thumb styles. Given that the birth year is a continuous variable, you could create a discrete variable for classification by doing appropriate segmentations (e.g., 1874-1900: 1, 1901-1920: 2, 1921-1940: 3). How does it affect the evaluation when you include the region while training the model? Now do the same using random forest (on both the above cases) and report the difference.

### 0. Set up
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv('blues_hand.csv')
```
<img width="602" alt="Screenshot 2024-02-13 at 8 56 23 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/ed4a49e9-3d38-4296-8819-54e1be9715f0">

### 1. Data Setting
(1) Cut birth_year into different year segments
```
bins = [0, 1901, 1921, np.inf]
names = ['1', '2', '3']

df['year_seg'] = pd.cut(df['brthYr'], bins, labels=names)

print(df.dtypes)
df
```
<img width="694" alt="Screenshot 2024-02-13 at 8 56 56 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/9b7805e9-d6ad-4361-b981-dd01a10895a9">

(2) Convert categorical data into int representations of unique categories
```
for col in df.columns:
  labels, uniques = pd.factorize(df[col])
  df[col] = labels
```
(3) Define variables and split dataset into training and testing datasets.
```ruby
X = df[['handPost', 'thumbSty']]
y = df['year_seg']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
```
(4) Data training
```ruby
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```
<img width="176" alt="Screenshot 2024-02-13 at 9 24 02 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/c68401ef-f814-46f1-9b66-488cbe9846be">

(5) Change variables and train again
```ruby
X2 = df[['region']]
y2 = df['year_seg']
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,test_size=0.3)
```
```ruby
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X2_train, y2_train)

predictions = rfc.predict(X2_test)
print(accuracy_score(y2_test, predictions))
print(confusion_matrix(y2_test, predictions))
```
<img width="170" alt="Screenshot 2024-02-13 at 9 24 13 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/85685730-df2d-4504-82b4-79f2f05e6ad6">

## Naive Bayes
A Naïve Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a piece of fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple. The assumption of independence is why it is known as naïve. It turns out that in most cases, even when such a naïve assumption is found to be not true, the resulting classification models do amazingly well.

We can start the Naive Bayes by classifying the YouTube spam collection dataset. This is a public set of comments collected for spam research. It has five datasets composed by 1,956 real messages extracted from five videos. These 5 videos are popular pop songs that were among the 10 most viewed on the collection period.

#### Variables
```
COMMENT_ID:  Unique id representing the comment
AUTHOR:      Author id,
DATE:        Date the comment is posted,
CONTENT:     The comment,
TAG:         For spam 1, otherwise 0
```
We can build a spam filter with Naive Bayes approach and use that filter to check the accuracy on the remaining dataset. 

### 0. Set up
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
```
```
df = pd.read_csv('Youtube02-KatyPerry.csv')
df1 = pd.read_csv('Youtube03-LMFAO.csv')
df2 = pd.read_csv('Youtube04-Eminem.csv')
df3 = pd.read_csv('Youtube05-Shakira.csv')
df4 = pd.concat([df, df1, df2, df3])
df4
```
<img width="1139" alt="Screenshot 2024-02-13 at 9 19 38 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/eca6359f-7ce9-4af5-86ec-2ee1061e91fb">

```
for col in df.columns:
  labels, uniques = pd.factorize(df4[col])
  df4[col] = labels
```
```
y = df4['CLASS']
X = df4.drop(columns='CLASS')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```
Assuming multinomial distribution
```
nb_multi = MultinomialNB()
nb_multi.fit(X_train,y_train)
```
```
from sklearn.metrics import accuracy_score, confusion_matrix
multi_preds = nb_multi.predict(X_test)
print(accuracy_score(y_test, multi_preds))
print(confusion_matrix(y_test, multi_preds))
```
<img width="172" alt="Screenshot 2024-02-13 at 9 22 37 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/8be5add7-f865-463e-944f-6936d5e02c14">

Assuming gaussian distribution
```
nb_gauss = GaussianNB()
nb_gauss.fit(X_train, y_train)

gauss_preds = nb_gauss.predict(X_test)
print(accuracy_score(y_test, gauss_preds))
print(confusion_matrix(y_test, gauss_preds))
```
<img width="168" alt="Screenshot 2024-02-13 at 9 22 52 PM" src="https://github.com/kellyhuanyu/Machine-Learning-Classification/assets/105426157/e45286a3-e66a-4bbb-baf9-3e69cff90208">
