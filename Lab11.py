'''
Lab 11
'''
print ("Lab 11")

##########Part 0 ###########
print("##########  PART 0  ##########\n")
'''
    1)  from sklearn.datasets import load_digits  (Each datapoint is a 8x8 image of a digit)
    breifly explain what are the features in this classification problem and how many features do we have?
    Find the distribution of the lables. 
    Use plot command to visualize the first five samples in the dataset. What are their lables?
    Split your data into train(80% of data) and test(20% of data) via random selection
'''
# So we have 64 features, from an 8 by 8 matrix. Inside each feature is a shade of black to white with 0 being black, and 16 being white. 

# YOUR CODE GOES HERE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()

best_overall_score = []
best_model = ['Logistic Regression', 'Linear SVC', 'SVC']

# the img as a array 8x8
plt.gray() 
for number in range(5):
    print(digits.images[number],"\n")
    plt.matshow(digits.images[number])

# plt.show()
    
# x_train, x_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=0.20)



##########Part 1 ###########
print("##########  PART 1  ##########\n")
'''
    1)  Try LogisticRegression from sklearn.linear_model
        Try to tune the hyperparameters (only change these params: penalty, C, Solver) via 5-Fold CV.
        What is the class_weight param? Do you need to modify that? Why?
'''
print("----------  Question 1  ----------\n")
# YOUR CODE GOES HERE

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

X = digits.data
y = digits.target

kf = KFold(n_splits=5)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
# Loop through C: 0.0 -> 20.0 incr by .1

def getScore( pen, solv, c, x_train, y_train ):
    try:
        logisticRegr = LogisticRegression(penalty=pen, solver=solv, C=c)
        logisticRegr.fit(x_train, y_train)
        return (logisticRegr.score(x_test, y_test))
    except:
        return 0

allScores = []
bestPen = ''
bestSolv = ''
bestC = ''
bestScore = 0

count = 0.0
while ( count != 5.0 ):
    print(count)
    for p in penalties:
        for  s in solvers:
            allScores.append(getScore(p, s, count, x_train, y_train))
            if max(allScores) > bestScore:
                bestScore = max(allScores)
                bestPen = p
                bestSolv = s
                bestC = count
    count += 1
print("Logistic Regression:")
print("\nBest score:", bestScore, "Using: pen: ", bestPen, " solv: ", bestSolv, " count: ", bestC, '\n')
best_overall_score.append(bestScore)

#  Through observation, we can see that the left most side and right most side of the number vectors are never used or are always zero,
#  so using a unified weighed would be best just in case their is an abnormal number that is skewed or too wide making the positions on the
#  sides of zeros 'weigh' too heavily on the decision of its target. 

'''
    2)  Try LinearSVC from sklearn.svm
    Try to tune the hyperparameters (only change these params: penalty, C, loss) via 5-Fold CV.
'''
print("----------  Question 2  ----------\n")
# YOUR CODE GOES HERE
from sklearn.svm import LinearSVC

penalties = [ 'l1', 'l2' ]
loss= ['hinge', 'squared_hinge']

def getScoreLinearSCV( pen, los, c, x_train, y_train, x_test, y_test ):
    try:
        linearSCV = LinearSVC(penalty=pen,loss=los, C=c)
        linearSCV.fit(x_train, y_train)
        return (linearSCV.score(x_test, y_test))
    except:
        return 0

allScores = []
bestPen = ''
bestLos = ''
bestC = ''
bestScore = 0

count = 0.0
while ( count != 10.0 ):
    print(count)
    for p in penalties:
        for  l in loss:
            allScores.append(getScoreLinearSCV(p, l, count, x_train, y_train, x_test, y_test))
            if max(allScores) > bestScore:
                bestScore = max(allScores)
                bestPen = p
                bestLos = l
                bestC = count
    count += 1
print("Linear SVC")
print("\nBest score:", bestScore, "Using: pen: ", bestPen, " loss: ", bestLos, " count: ", bestC, '\n')
best_overall_score.append(bestScore)

'''
    3)  Try SVC from sklearn.svm (this classifier can also be used with linear kernel == LinearSVC)
    Try to tune the hyperparameters (only change these params: decision_function_shape, C, kernel, degree) via 5-Fold CV.
'''
# YOUR CODE GOES HERE
from sklearn.svm import SVC

# c = float 0.0 -> 5.0
kernel= ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed', 'callable' ]
# degree = int 1-5
func_shape = [ 'ovo', 'ovr' ]
 
def getScoreSVC( k, d, f, c, x_train, y_train, x_test, y_test ):
    try:
        svc = SVC( C=c, kernel=k, degree=d, decision_function_shape=f)
        svc.fit(x_train, y_train)
        return (svc.score(x_test, y_test))
    except:
        return 0

allScores = []
bestK = ''
bestD = 0
bestC = 0.0
bestF = ''

bestScore = 0

count = 0
degree_count = 0
while ( count != 10.0 ):
    print(count)
    for k in kernel:
        for  d in range (5):
            degree_count += 1
            for f in func_shape:    
                allScores.append(getScoreSVC(k, d, f, count, x_train, y_train, x_test, y_test))
                if max(allScores) > bestScore:
                    bestScore = max(allScores)
                    bestK = k
                    bestC = count
                    bestD = d
                    bestF = f
    count += .5
print("SVC")
print("\nBest score:", bestScore, "Using: Kernel: ", bestK, " C: ", bestC, " Degree: ", bestD, 'Funct_Shape:', bestF,'\n')
best_overall_score.append(bestScore)


##########Part 2 ###########
print("##########  PART 2  ##########\n")
'''
    1)  Test your trained models in part1: Q1, Q2, and Q3 with the test set and pick the best model. Try to analyze the confusion matrix and explain which classes are mostly confused with each other.
'''
# YOUR CODE GOES HERE
print("The best one is:", best_model[best_overall_score.index(max(best_overall_score))])

##########Part 3 ###########
# IGNORE
print("##########  PART 3  ##########\n")
'''
    1)  Repeat part 1 and 2 with Normalized data
'''
# YOUR CODE GOES HERE

