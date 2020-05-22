# Adam Capdeville
# Isaac Garza

# ========== Question 1 ========== #

### Part A: ###
'''
Briefly explain what this method is.
Explanation:
    Principal Component Analysis (PCA) is a technique for extracting the most important features from a dataset. 
    PCA does this through linear dimensionality reduction using Singular Value Decomposition (SVD) of the data 
    to project it to a lower dimensional space. PCA reduction allows us to easily explore and visualize data.
'''

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

def pca_func(n):
    ### Part B: ###
    '''
    From sklearn.datasets import load-digits (Each data point is a 8x8 image of a digit (64 features)). 
    Split your data into train(80% of data) and test(20% of data) via random selection.
    '''
    (data, target) = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)


    ### Part C: ###
    '''
    Use PCA from sklearn package to reduce the dimensionality of this dataset to 10.
    '''



    pca = PCA(n_components=n)
    pca.fit(x_train)
    x_train_proj = pca.transform(x_train)
    x_test_proj = pca.transform(x_test)

    print("\nOriginal Data shape:", x_train.shape)
    print("x_train_proj shape:", x_train_proj.shape)
    print("x_test_proj shape: ", x_test_proj.shape)

    ### Part D: ###
    '''
    Print ”explained-variance-ratio-” and explain what does it represent?

    Explanation: 
        ( pca.explained_variance_ratio_ ): returns a vector of the varaince explained by each dimension.
    '''
    print("\nExplained variance ratio (for ",n,"components):")
    print(pca.explained_variance_ratio_, "\n")

    ### Part E: ###
    '''
    Train a Logistic Regression model for the original data, and transformed data. compare
    the results. Explain why they are different.
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score


    clf = LogisticRegression(random_state=42)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)

    # print(5*'\n')

    clf2 = LogisticRegression(random_state=42)
    clf2.fit(x_train_proj, y_train)
    predicted2 = clf2.predict(x_test_proj)

    # generate evaluation metrics
    accuracy = (accuracy_score(y_test, predicted))
    accuracy2 = (accuracy_score(y_test, predicted2))

    print("Regular Data Accuracy: ",accuracy)
    print("Transformed Data Accuracy: ", accuracy2)


### PART F: ###
'''
Repeat (c) ,(d) ,and (e) with 5 components.
'''

n = int(input("Please give me n_components: "))
pca_func(n)

### PART G: ###
'''
Compare the results of your trained classifiers with 10 and 5 components. Explain why
they are different.

Explanation:
    When we set PCA to extract and train only the first 10 Principal Components (P.C.) we get
    an Accuracy of .911111 (1's repeating), for 5 P.C. we get .838888 (8 repeating). Our accuracys 
    readings are weaker with fewer P.C. because the explained vairance when dimensionaly reduced is 
    very low. Out of 64 pixels the first (and best) P.C. only explains .14 . This means our P.C. 
    explained variance is spread out across all 64 pixels. 
    
    ( Just for fun ): We increased the P.C. captured to 30 and got an accuracy of .9611 (1's repeating) 
    which is the same accuracy as the original data set. This means we use less than half of the original 
    64 pixels (features) to get the same percision. Which is pretty cool! 

'''