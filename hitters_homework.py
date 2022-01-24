import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import warnings
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("../input/hitters/Hitters.csv")
df = df_.copy()
df.head()

def check_df(dataframe, head=5, tail = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head ######################")
    print(dataframe.head(head))
    print("##################### Tail ######################")
    print(dataframe.tail(tail))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("################### Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df.nunique()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.
    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables
    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    import pandas as pd
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

num_summary(df, num_cols, True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Salary", col)

df.corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

correlation_matrix = df.corr().round(2)
threshold=0.75
filtre=np.abs(correlation_matrix['Salary']) > 0.50
corr_features=correlation_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(),annot=True,fmt=".2f")
plt.title('Correlation Between Features')
plt.show()
# See just 'Salary' feature has NaN values. Now, correlation that is what's going between features.
# How are they strict relation between them. We gave correlation values more than 0.5 between features.

msno.bar(df);
plt.show()

df.isnull().sum()
# There are 59 missing observation units in the "Salary" variable of our data.

def pairplot(dataset, target_column):
    sns.set(style="ticks")
    sns.pairplot(dataset, hue=target_column)
    plt.show()

pairplot(df, 'Salary')

df["New_Avg_CAtBat"] = df["CAtBat"] / df["Years"]
df["New_Avg_CHits"] = df["CHits"] / df["Years"]
df["New_Avg_CHmRun"] = df["CHmRun"] / df["Years"]
df["New_Avg_Cruns"] = df["CRuns"] / df["Years"]
df["New_Avg_CRBI"] = df["CRBI"] / df["Years"]
df["New_Avg_CWalks"] = cwalks = df["CWalks"] / df["Years"]

df["New_Hits/AtBat_Rate"] = (df["Hits"] / df["AtBat"])*100
df["New_HmRun/Hits_Rate"] = (df["HmRun"] / df["Hits"])*100
df["New_Runs/Hits_Rate"] = (df["Runs"] / df["Hits"])*100
df["New_RBI/Hits_Rate"] = (df["RBI"] / df["Hits"])*100
df["New_Errors/Walks_Rate"] = (df["Errors"] / df["Walks"])*100
df["New_CHits/CAtBat_Rate"] = (df["CHits"] / df["CAtBat"])*100
df["New_CHmRun/CHits_Rate"] = (df["CHmRun"] / df["CHits"])*100
df["New_CRuns/CHits_Rate"] = (df["CRuns"] / df["CHits"])*100
df["New_CRBI/CHits_Rate"] = (df["CRBI"] / df["CHits"])*100

df["New_Hit_Success"] = df["CHits"]/df["CAtBat"]

df["New_Rating"] = (24*df["HmRun"] + 21*df["Runs"] + 21*(df["Walks"] - df["Errors"]) + 18*df["Assists"] + 16*df["RBI"])/100

df["New_Best"]=df["Hits"]*df["CRBI"]*df["PutOuts"]

df.head()

df["Salary"].fillna(df["Salary"].median(), inplace=True)

df.isnull().sum()

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))
# When we rechecked the variables with outlier values, we detected outlier values in the variables.
# Variables with outlier values are:

# "New_Hits/AtBat_Rate",
# "New_Runs/Hits_Rate",
# "New_Errors/Walks_Rate",
# "New_CHits/CAtBat_Rate",
# "New_Hit_Success"
# "New_Best"

for col in num_cols:
    replace_with_thresholds(df, col)
# We suppressed outliers according to the lower and upper bounds of the threshold values.

for col in num_cols:
    print(col, check_outlier(df, col))
# As can be seen, no threshold value problem was observed.

sns.boxplot(x = df["Salary"]);
plt.show()

q3 = 0.90
salary_up = int(df["Salary"].quantile(q3))
df = df[(df["Salary"] < salary_up)]

sns.boxplot(x=df["Salary"])
plt.show()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    labelencoder = LabelEncoder()
    df[col] = labelencoder.fit_transform(df[col])

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=False)
    return dataframe

df = one_hot_encoder(df, cat_cols)

scaler = RobustScaler()
num_cols = [col for col in num_cols  if col not in ["Salary"]]
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop("Salary", axis=1)
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression().fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
reg_model.score(X_train, y_train)
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))

pre_model = LGBMRegressor().fit(X, y)
feature_imp = pd.DataFrame({'Feature': X.columns, 'Value': pre_model.feature_importances_})
feature_imp.sort_values("Value", ascending=False)

pre_model = RandomForestRegressor().fit(X, y)
feature_imp = pd.DataFrame({'Feature': X.columns, 'Value': pre_model.feature_importances_})
feature_imp.sort_values("Value", ascending=False)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
