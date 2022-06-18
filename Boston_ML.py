""" This is a basic ML problem on prediction of Bosteon house price using decision tree model"""

""" Any database can be implemented for regression and models can be tuned for improving the performance"""

# =============================================================================
#         Prediction of price of Boston house database
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    
    df = load_boston()    # Dictionary containing all information
    print (df.keys())
    
    df_boston = pd.DataFrame(df.data, columns=df.feature_names)    ## data stores all features
    df_boston["Price"] = df.target           ## adds a column name Price, which is the target property
    
    # features and target
    X = df_boston.iloc[:,:-1]                # features
    Y = df_boston.iloc[:,-1]                 # target property
    
    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.1) 
    
    #model train and fit
    model = DecisionTreeRegressor(random_state=1)
    model.fit(X_train,Y_train)
    
    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)
    
    # prediction results
    print ("R2 (train/test) = ", r2_score(Y_train,predict_train), "/", r2_score(Y_test,predict_test))
    print ("MSE (train/test) = ", mean_squared_error(Y_train,predict_train), "/", mean_squared_error(Y_test,predict_test))
    
    # parity plot of model
    fig, ax = plt.subplots()
    ax.scatter(Y_test, predict_test, edgecolors=(0, 0, 0))
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Truth vs Predicted")
    plt.show()

if __name__ == "__main__":
    main()
