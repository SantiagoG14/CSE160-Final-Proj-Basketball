# %%
# imports
import pandas as pd
import numpy as np
import plotly.express as pltx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate


# %%
# datasets
dolt_players = pd.read_csv("datasets/dolthub_SHAQ_main_players.csv")
players = pd.read_csv("datasets/player_data.csv")
data = pd.read_csv("datasets/dolthub_SHAQ_main_player_season_stat_totals.csv")
# clean data
data.dropna(inplace=True)


# %% [markdown]
# # Goals
# * Regression: Predict future statistics of NBA players based on past performance
# 

# %%
# Regression: predicting future statistics
# Start with a linear regression model
# Incorporate Lasso

# seperating x from y
def split_data(df):
    temp = df.drop(columns={"player_id","season_id", "team_id","season_type_id","league_id"})
    x = temp.drop(columns={"minutes"})
    y = temp["minutes"]
    return (x, y)

def split_data_test_train(x, y):
    return train_test_split(x, y, test_size=.2, random_state=23)


# %%
unique = len(data["player_id"].unique())
total_min = data["minutes"].sum()
print(total_min / unique)


# %%
# shaq
x, y = split_data(data)
x_train, x_test, y_train, y_test = split_data_test_train(x, y)

def calculate_mse(model, x_test=x_test, y_test=y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred

# %%
def create_plot(pred, title):
    fig = pltx.scatter(data,
            x=y_test,
            y=pred,
            trendline="ols",
            trendline_color_override="white",
            color=pred,
            template="plotly_dark",
            title=title + " to Predict Minutes Played",
            labels={
                "y":"Predicted Minutes Played",
                "x":"Actual Minutes Played",
                "color":"Legend"
            }
            )
    return fig

# %%
lr = LinearRegression()
lr.fit(x_train, y_train)
mse, pred = calculate_mse(lr, x_test, y_test)
print(mse)
create_plot(pred, "Linear Regression")

# %%
# lasso
lasso = Lasso(alpha=0.1, tol=.001)
lasso.fit(x_train, y_train)
mse, pred = calculate_mse(lasso, x_test, y_test)
print(mse)
create_plot(pred, "Lasso Regression")

# %% [markdown]
# Can we use other models to predict?
# * Decision Tree Regression
# * K nearest neighbor
# * Random Forest

# %%
# Decision Tree Regression
tree = DecisionTreeRegressor(max_depth=10)
tree.fit(x_train, y_train)
mse, pred = calculate_mse(tree, x_test, y_test)
print(mse)
create_plot(pred, "Decision Tree Regression")

# %%
# K nearest neighbors
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)
mse, pred = calculate_mse(knn, x_test, y_test)
print(mse)
create_plot(pred, "K Nearest Neighbors Regression")

# %%
# Random Forest
random_forest = RandomForestRegressor(n_estimators=10)
random_forest.fit(x_train, y_train)
mse, pred = calculate_mse(random_forest, x_test, y_test)
print(mse)
create_plot(pred, "Random Forest Regression")

# %%
from sklearn.metrics import make_scorer

def return_mse(y_p, y_t):
    return mean_squared_error(y_p,y_t)

def cross_val(model, x=x_train, y=y_train,cv=5):
    scorer = make_scorer(return_mse)
    results = cross_validate(estimator=model, X=x, y=y, cv=cv, scoring=scorer)
    return results["test_score"].mean()


accuracy = {}
for model in [lr, lasso, tree, knn, random_forest]:
    accuracy[model] = cross_val(model)
print(accuracy)

# %%
# optimize features
def optimize_features(data, coef):
    temp = data.drop(columns={"player_id","season_id","team_id","minutes","season_type_id","league_id"})
    mask = np.abs(coef) > 0
    optimized = temp.loc[:, mask]
    return optimized

# %%
# What are the greatest features?
coefficients = pd.DataFrame({"Coefficients":x_train.columns, "Values":lasso.coef_})
# coefficients_sorted = coefficients.sort_values(by='Values', ascending=False)
# print(coefficients_sorted)
print(coefficients)

# %% [markdown]
# # The most important factors are games played and starting

# %%
x_new = optimize_features(data, lasso.coef_)
y_new = data["minutes"]
new_x_tr, new_x_te, new_y_tr, new_y_te = split_data_test_train(x_new, y_new)
random_forest.fit(new_x_tr, new_y_tr)
mse, pred = calculate_mse(random_forest, new_x_te, new_y_te)

# %%
print(mse)
create_plot(pred, "Random Forest without Games Played or Started")

# %%
cross_val(random_forest)


# %%



