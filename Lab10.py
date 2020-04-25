
'''
Lab 10
'''
print ("Lab 10")

def part3(norm):

    ##########Part 0 ###########
    print("\n-------------------------------- PART 0 --------------------------------\n")
    '''
        1)  from sklearn.datasets import load_boston
        Extract the description of all the features and print it
        Split your data into train(80% of data) and test(20% of data) via random selection
    '''
    print("\n--------------------------------\n\t\t\tQ0\n--------------------------------\n")
    # YOUR CODE GOES HERE
    
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    data, target = load_boston(return_X_y=True)

    if norm == True:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler() 
        data = scaler.fit_transform(data)
        """
          elif selection == 'minmax':
        min_max_scaler = MinMaxScaler().fit(data) # might have to fit it like in standarad
        return min_max_scaler.transform(data)
        """
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)
    else:
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)

    print(data.shape)
    print(target.shape)



    ##########Part 1 ###########
    print("\n-------------------------------- PART 1 --------------------------------\n")
    '''
        1)  Try LinearRegression from sklearn.linear_model   
            Try it with and without normalization. Compare the results and pick the best trained model(for comparisson try different metrics from sklearn.metrics like: r2, mse, mae)
            (Hint: for normalizing your data set normalize=True)
        
    '''
    print("\n--------------------------------\n\t\t\tQ1\n--------------------------------")
    # YOUR CODE GOES HERE
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    all_score = []

    reg = LinearRegression(normalize=False).fit(x_train, y_train)

    y_pred = reg.predict(x_test)

    results = ['r2', 'mae', 'mse']
    scores = []
    scores.append(r2_score(y_test, y_pred))
    scores.append(mean_absolute_error(y_test, y_pred))
    scores.append(mean_squared_error(y_test, y_pred))

    print("Linear Regression")
    print("Metrics: ",results[scores.index(min(scores))], " with a score of:",min(scores))

    all_score.append(min(scores))
    
    '''
        2)  Try DecisionTreeRegressor from sklearn.tree
        Compare the results and pick the best trained model. 
    '''
    print("\n--------------------------------\n\t\t\tQ2\n--------------------------------")
    # YOUR CODE GOES HERE
    from sklearn.tree import DecisionTreeRegressor
    # criterion {“mse”, “friedman_mse”, “mae”}, default=”mse”

    dtr = DecisionTreeRegressor()
    dtr_friedman = DecisionTreeRegressor(criterion="friedman_mse")
    dtr_mae = DecisionTreeRegressor(criterion="mae")

    dtr.fit(x_train, y_train)
    dtr_friedman.fit(x_train, y_train)
    dtr_mae.fit(x_train, y_train)


    y_pred_mse = dtr.predict(x_test)
    # # print("MSE: \n", y_pred_mse)
    # mse_score = dtr.score(x_test, y_pred_mse)
    mse_score = mean_squared_error(y_test, y_pred_mse)
    # print("MSE SCORE: ", mse_score)


    y_pred_friedman = dtr_friedman.predict(x_test)
    # print("Friedman MSE: \n", y_pred_friedman)
    # friedman_score = dtr_friedman.score(y_test, y_pred_friedman)
    friedman_score = mean_squared_error(y_test,y_pred_friedman)
    # print("Friedman SCORE: ", friedman_score)


    y_pred_mae = dtr_mae.predict(x_test)
    # print("MAE: \n", y_pred_mae)
    # mae_score = dtr_mae.score(y_test, y_pred_mae)
    mae_score = mean_absolute_error(y_test,y_pred_mae)
    # print("MAE SCORE: ", mae_score)

    results = ['mse', 'friedman_mse', 'mae']
    scores = []
    scores.append(mse_score)
    scores.append(friedman_score)
    scores.append(mae_score)

    # print("The best score being:", min(scores), "From :", results[scores.index(min(scores))])
    print("Decision Tree Regressor")
    print("Criterion: ",results[scores.index(min(scores))], " with a score of:",min(scores))
    all_score.append(min(scores))
    '''
        3)  Try KNeighborsRegressor from sklearn.neighbors
        Try it for different values of n_neighbors and different types of algorithm. Compare the results and pick the best trained model.s
    '''
    print("\n--------------------------------\n\t\t\tQ3\n--------------------------------")
    # YOUR CODE GOES HERE
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import accuracy_score
    # from sklearn.model_selection import cross_val_score

    # print("length of x_train:", len(x_train), " len of x_test:", len(x_test))
    # print("length of y_train:", len(y_train), " len of y_test:", len(y_test))


    def FIND_KNR_PRED(n_neighbors, method, x_train, y_train, x_test):
        knn = KNeighborsRegressor(n_neighbors= n_neighbors, algorithm=method)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        return y_pred


    algorithms = ["ball_tree", "kd_tree", "brute"]
    results = ['r2', 'mae', 'mse']
    result_scores = [0,0,0]

    n_neigh = 5

    for n in range(1,n_neigh):
        for alg in algorithms:
            scores = []
            knr_pred = FIND_KNR_PRED( n, alg, x_train, y_train, x_test)
            scores.append(r2_score(y_test, knr_pred))
            scores.append(mean_absolute_error(y_test, knr_pred))
            scores.append(mean_squared_error(y_test, knr_pred))
            result_scores[scores.index(min(scores))] += 1

    final_result = results[result_scores.index(max(result_scores))]
    print("K Neighbors Regressor")
    print("Metric: ", final_result, " gives us the best error.")
    
    knr_pred = FIND_KNR_PRED( n, "auto", x_train, y_train, x_test)

    if(final_result == 'r2'):
        print("R2 Score:", r2_score(y_test, knr_pred))
        all_score.append(r2_score(y_test, knr_pred))
    elif(final_result == 'mae'):
        print("MSE Score:", mean_absolute_error(y_test, knr_pred))
        all_score.append(mean_absolute_error(y_test, knr_pred))
    else:
        print("MAE Score:", mean_squared_error(y_test, knr_pred))
        all_score.append(mean_squared_error(y_test, knr_pred))


    ##########Part 2 ###########
    print("\n-------------------------------- PART 2 --------------------------------\n")

    '''
        1)  Test your trained models in part1: Q1, Q2, and Q3 with the test set and pick the best model
    '''
    # YOUR CODE GOES HERE
    model_list = ["LinearRegression", "DecisionTreeRegressor", "KNeighborsRegressor"]
    print(model_list)
    print(all_score)
    print("Based off our Metrics, Criterion & Algorithm results : ", model_list[all_score.index(min(all_score))],"\n")

    


##########Part 3 ###########
'''
    1)  Repeat part 1 and 2 with Normalized data
'''
print("\n-------------------------------- PART 3 --------------------------------\n")
# YOUR CODE GOES HERE

answer = input("Would you like to normalize the data?, Y / N: ")

if (answer == "Y"):
    part3(True)
else:
    part3(False)


