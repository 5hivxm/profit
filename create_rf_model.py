import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Calculate stats based on input data
def calculate_stats(data):
    data['Profit'] = (data['Price'] - data['Cost'])*data['Demand']
    data['PriceDiff'] = data['Price'] - data['CompetitorPrice']
    data['Markup'] = (data['Price'] - data['Cost']) / data['Cost']
    return data

# Creating mapping for categorical values
def mappings(data, brand_map, product_map):
    data['Brand'] = data['Brand'].map(brand_map)
    data['Product'] = data['Product'].map(product_map)
    return data

# Price optimization function
def calculate_profit(item_data, price_range, model, og_demand, graph_data):
    best_price = item_data['Price']
    max_profit = item_data['Profit']
    best_demand = og_demand
    
    for price in price_range:
        item_data['Price'] = price
        cost = item_data['Cost']
        item_data['PriceDiff'] = item_data['Price'] - item_data['CompetitorPrice']
        item_data['Markup'] = (price - cost) / cost
        demand = model.predict(pd.DataFrame([item_data]))[0]
        profit = (price - cost) * demand

        if profit > max_profit:
            max_profit = profit
            best_price = price
            best_demand = demand

        graph_data[price] = profit

    return best_demand, best_price, max_profit, graph_data

# Reverse mapping to return brand, product names
def reverse_stats(ex, brand_map, product_map):
    reverse_brandmap = {number: brand for brand, number in brand_map.items()}
    ex['Brand'] = ex['Brand'].map(reverse_brandmap)
    reverse_prodsmap = {number: prods for prods, number in product_map.items()}
    ex['Product'] = ex['Product'].map(reverse_prodsmap)
    return ex

# Track changes in features
def changes(full):
    original_price = sum(full['Original Price'])
    optimal_price = sum(full['Optimal Price'])
    original_profit = sum(full['Original Profit'])
    optimal_profit = sum(full['Max Profit (Optimal Price)'])
    original_demand = sum(full['Original Demand'])
    optimal_demand = sum(full['Optimal Demand'])
    original_revenue = sum(full['Original Price'] * full['Original Demand'])
    optimal_revenue = sum(full['Optimal Price'] * full['Optimal Demand'])
    demands = ['Demand', int(round(original_demand)), int(round(optimal_demand)), round((optimal_demand - original_demand) / original_demand * 100, 2)]
    revs = ['Revenue', round(original_revenue,2), round(optimal_revenue,2), round((optimal_revenue - original_revenue) / original_revenue * 100, 2)]
    profits = ['Profit', round(original_profit,2), round(optimal_profit,2), round((optimal_profit - original_profit) / original_profit * 100, 2)]
    return demands, revs, profits

def main():

    df = pd.read_csv('luxury_real_data.csv')
    brand_map = {brand: index for index, brand in enumerate(df['Brand'].unique())}
    product_map = {products: index for index, products in enumerate(df['Product'].unique())}
    df = calculate_stats(df)
    df = mappings(df, brand_map, product_map)

    # Define features and target
    y = df['Demand']
    X = df.drop(['Demand', 'Competitor'], axis=1)
    X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)

    # Initial Random Forest Regression
    rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

    # Hyperparameter Tuning using RandomizedSearchCV
    n_estimators = [int(x) for x in np.linspace(start = 0, stop = 1000, num = 100)]
    max_features = [1.0, 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 31, num = 6)]
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

    rf_ransearch = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                n_iter = 10, scoring='neg_mean_squared_error',
                                cv = 5, verbose=2, random_state=42, n_jobs=-1).fit(X_train,y_train)
    
    # Random Forest Regerssion with optimized hyperparameters
    rf = RandomForestRegressor(**rf_ransearch.best_params_).fit(X_train, y_train)

    filename = "random_forest.pickle"
    pickle.dump(rf, open(filename, "wb"))

    st.header('done')


if __name__ == '__main__':
    st.title("Optimal Price for Luxury Fashion Brands")
    main()