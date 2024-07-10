import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Modeling Low Volume Transactoins")
st.subheader(
    "Sales Dataset"
)


# Universal
data = pd.read_csv('price_optimization_dataset.csv')
y = data['Demand']
X = data.drop(['ProductID', 'Demand'], axis=1)
best_models = []
X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)
st.dataframe(data)


# Training model on training set
# Predicting Demands
rf_train = RandomForestRegressor(n_estimators=100, random_state=42)
rf_train.fit(X_train, y_train)
best_models.append(['rf_train', rf_train.score(X_train, y_train),
                    mean_squared_error(y_val,rf_train.predict(X_val)),
                    rf_train.score(X_val, y_val)])


# Training on combined training+validation set
X_train_val = pd.concat([X_train,X_val])
y_train_val = pd.concat([y_train,y_val])
rf_final = RandomForestRegressor(random_state=42)
rf_final.fit(X_train_val,y_train_val)
best_models.append(['rf_final', rf_final.score(X_train_val, y_train_val),
                    mean_squared_error(y_test,rf_final.predict(X_test)),
                    rf_final.score(X_test, y_test)])

preds = rf_final.predict(X)
pre_profit = sum(data['Profit'])
pre_demand = sum(y)


# RandomSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1200, num = 10)]
max_features = [1.0, 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,            # number of trees
               'max_features': max_features,            # max splitting node features
               'max_depth': max_depth,                  # max levels in each tree
               'min_samples_split': min_samples_split,  # min data in a pre-split node
               'min_samples_leaf': min_samples_leaf,    # min data allowed in leaf
               'bootstrap': bootstrap}                  # replacement or not

rf_ransearch = RandomizedSearchCV(estimator=rf_final, param_distributions=random_grid,
                              n_iter = 10, scoring='neg_mean_absolute_error',
                              cv = 3, verbose=2, random_state=42, n_jobs=-1)

rf_ransearch.fit(X_train_val,y_train_val)

# GridSearchCV
# Using best parameters from RandomSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [None, 10, 20],
    'max_features': [1],
    'min_samples_leaf': [1],
    'min_samples_split': [5, 7],
    'n_estimators': [600, 700, 800]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf_final, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_val,y_train_val)
best_models.append(['rf_gridsearch', grid_search.score(X_train_val, y_train_val),
                    mean_squared_error(y_test,grid_search.predict(X_test)),
                    grid_search.score(X_test, y_test)])

best_models = pd.DataFrame(best_models, columns=['Model', 'Train Score', 'MSE', 'Test/Validation Score'])
st.subheader(
    "Random Forest Model Comparison"
)
st.table(best_models)


# Predicting Price
temp = data.copy()
X_prices = pd.DataFrame(preds)
y_prices = temp['Price']
X_1, X_2, y_1, y_2 = train_test_split(X_prices, y_prices, test_size=0.2, random_state=42)

model = ElasticNet(alpha=0.1,l1_ratio=0.5)
model.fit(X_1, y_1)
prices = model.predict(X_prices)


# Price optimization function for one instance
def find_results(graph_data, item_data, price_range, model):
  best_price = None
  max_profit = -np.inf

  for price in price_range:
      item_data['Price'] = price
      cost = item_data['ProductionCost']
      item_data['PriceDiff'] = item_data['Price'] - item_data['CompetitorPrice']
      item_data['Markup'] = (price - cost) / cost
      demand = model.predict(pd.DataFrame([item_data]))[0]
      profit = (price - cost) * demand
      graph_data.append([price, demand, profit])

      if profit > max_profit:
          max_profit = profit
          best_price = price

  return demand, best_price, max_profit

# Test the optimization for dataset
results = []
graph_data = []
i=0
sample_item = X.iloc[i].copy()
sample_price = prices[i]
price_range = np.linspace(sample_item['Price'], sample_price, 100)
opt_demand, opt_price, max_profit = find_results(graph_data, sample_item, price_range, rf_final)
results.append([opt_demand, opt_price, max_profit])

results = pd.DataFrame(results, columns=['Optimal Demand', 'Optimal Price', 'Max Profit'])
graph_data = pd.DataFrame(graph_data, columns=['Price', 'Demand', 'Profit'])

# Plotting Example Data
fig, ax = plt.subplots()
sns.lineplot(data=graph_data, x="Price", y="Profit", ax=ax)
sns.scatterplot(x='Optimal Price', y='Max Profit', data=results, s=50, color='red', ax=ax)
ax.set_title('Optimal Price for Profit (Item 0)')
ax.set_xlabel('Price')
ax.set_ylabel('Profit')
ax.legend()
st.pyplot(fig)


# Optimizing Profit
# Price optimization function
def find_results(item_data, price_range, model):
  best_price = None
  max_profit = -np.inf

  for price in price_range:
      item_data['Price'] = price
      cost = item_data['ProductionCost']
      item_data['PriceDiff'] = item_data['Price'] - item_data['CompetitorPrice']
      item_data['Markup'] = (price - cost) / cost
      demand = model.predict(pd.DataFrame([item_data]))[0]
      profit = (price - cost) * demand

      if profit > max_profit:
          max_profit = profit
          best_price = price

  return demand, best_price, max_profit

# Test the optimization for dataset
results = []

for i in range(len(X)):
  sample_item = X.iloc[i].copy()
  sample_price = prices[i]                # Using predicted price as max/min price in price range
  price_range = np.linspace(sample_item['Price'], sample_price, 100)
  opt_demand, opt_price, max_profit = find_results(sample_item, price_range, rf_final)
  results.append([opt_demand, opt_price, max_profit])

results = pd.DataFrame(results, columns=['Optimal Demand', 'Optimal Price', 'Max Profit'])
st.subheader(
    "Dataset of Optimal Demand, Prices for Maximum Profit"
)
st.dataframe(results)

# Displaying Comparison Table
full = pd.merge(results, data[['Demand', 'Price', 'Profit']], left_index=True, right_index=True)
columns = ['Optimal Demand', 'Optimal Price', 'Max Profit', 'Original Demand', 'Original Price', 'Original Profit']
full.columns = columns
full = full[['Original Demand', 'Optimal Demand', 'Original Price', 'Optimal Price', 'Original Profit', 'Max Profit']]
full[['Original Demand', 'Optimal Demand']] = full[['Original Demand', 'Optimal Demand']].round()
full[['Original Price', 'Optimal Price', 'Original Profit', 'Max Profit']] = full[['Original Price', 'Optimal Price', 'Original Profit', 'Max Profit']].round(2)

# Displaying Percent Changes and Elasticities
original_price = sum(full['Original Price'])
optimal_price = sum(full['Optimal Price'])
original_profit = sum(full['Original Profit'])
optimal_profit = sum(full['Max Profit'])
original_demand = sum(full['Original Demand'])
optimal_demand = sum(full['Optimal Demand'])
original_revenue = sum(full['Original Price']*full['Original Demand'])
optimal_revenue = sum(full['Optimal Price']*full['Optimal Demand'])

demands = ['Demand', original_demand, optimal_demand, round((optimal_demand-original_demand)/original_demand*100, 2),
           (optimal_demand-original_demand)/(optimal_price - original_price)]
revs = ['Revenue', original_revenue, optimal_revenue, round((optimal_revenue - original_revenue)/original_revenue*100, 2),
        (optimal_revenue-original_revenue)/(optimal_price - original_price)]
profits = ['Profit', original_profit, optimal_profit, round((optimal_profit-original_profit)/original_profit*100, 2),
           (optimal_profit-original_profit)/(optimal_price - original_price)]

columns = ['Feature', 'Original', 'Optimized', 'Percent Change', 'Elasticity']
res = pd.DataFrame([demands, revs, profits], columns=columns)

st.subheader(
    "Full Dataset Comparing Demands, Prices, Profits"
)
st.dataframe(full)
# Plotting original vs optimal Profit
fig, ax = plt.subplots()
sns.scatterplot(x='Original Price', y='Original Profit', data=full, s=50, color='blue', ax=ax)
sns.scatterplot(x='Optimal Price', y='Max Profit', data=full, s=50, color='red', ax=ax)
ax.set_title('Price vs Profit')
ax.set_xlabel('Price')
ax.set_ylabel('Profit')
st.pyplot(fig)

st.subheader(
    "Increases in Demand, Revenue, Profit, Elasticities"
)
st.table(res)

