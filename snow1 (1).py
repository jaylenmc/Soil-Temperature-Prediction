import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xanfis import GdAnfisRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xanfis
import sklearn
import math
import xgboost

snow_csv = pd.read_csv('Grand Forks_daily updated (1).csv')

snow_csv['Time(CST)'] = pd.to_datetime(snow_csv['Time(CST)'], format='%m/%d/%Y')
snow_filtered = snow_csv[snow_csv['Time(CST)'].dt.month.isin([10,11,12, 1, 2, 3])]

snow_filtered.isnull().count()
snow_filtered=snow_filtered.fillna(0)
st_value_list = ['ST_10', 'ST_50', 'ST_100']

for st_value in st_value_list:
    train = snow_filtered.drop([other_st for other_st in st_value_list if other_st != st_value], axis=1)
    # train = snow_filtered.drop(['ST_50', 'ST_100', 'Time(CST)'], axis=1)
    #Y=snow_filtered['ST_10']

    # Correlation Matrix
    f, ax = plt.subplots(figsize=(10, 8))
    corr = train.corr()

    sns.heatmap(
        corr, 
        mask=np.zeros_like(corr, dtype=bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=True,
        fmt=".2f"
    )

    ax.set_title(f"Correlation Matrix for {st_value}")
    plt.show()

    scaler = MinMaxScaler()
    Scalable=[
        'AvgAirTemp_C', 
        'AvgRelHum_%', 
        'AvgWindSpeed_m/s', 
        'AvgWindDir_Deg', 
        'TotalSolRad_MJ/m2', 
        'Rainfall_mm', 
        'SnowDepth_mm', 
        'SoilMoisture'
        ]
    train[Scalable] = scaler.fit_transform(train[Scalable])
    X=train.drop([st_value, 'Time(CST)'], axis=1)
    Y=train[st_value]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=52)

    print(f"========== Fuzzy Model ({st_value}) ==========")
    # Fuzyy Logic Model
    model = GdAnfisRegressor(
        num_rules=20, 
        mf_class="Trapezoidal",
        act_output=None, 
        vanishing_strategy="blend", 
        reg_lambda=None,
        epochs=50, 
        batch_size=8, 
        optim="Adafactor", 
        optim_params={"lr": 0.01},
        early_stopping=False, 
        n_patience=10, 
        epsilon=0.001, 
        valid_rate=0.1,
        seed=42, 
        verbose=True
        )
    model.fit(X_train.values,Y_train.values)
    model_pred = model.predict(X_test.values)
    mse = sklearn.metrics.mean_squared_error(Y_test.values,model_pred)
    rmse = math.sqrt(mse)
    print(f"RMSE Value: {rmse}")

    print(f"========== Random Forest ({st_value}) ==========")
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=1,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        )
    rf.fit(X_train, Y_train)
    rf_pred = rf.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test,rf_pred)
    rmse = math.sqrt(mse)
    print(f"RMSE Value: {rmse}")

    print(f"========== XgBoost ({st_value}) ==========")
    # XgBoost
    xgboost_model = xgboost.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        gamma=1, 
        subsample=0.75,
        colsample_bytree=0.7, 
        max_depth=5
        )
    xg=xgboost_model.fit(X_train,Y_train)
    y_pred=xg.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test,y_pred)
    rmse = math.sqrt(mse)
    print(f"RMSE Value: {rmse}")

    print(f"========== Linear Regression ({st_value}) ==========")
    # Linear Regression
    lr = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
    lr.fit(X_train, Y_train)
    lr_pred = lr.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test,lr_pred)
    rmse = math.sqrt(mse)
    print(f"RMSE Value: {rmse}\n")