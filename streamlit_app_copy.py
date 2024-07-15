import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from scipy.optimize import minimize_scalar
import math

# Show app title and description
st.set_page_config(page_title="Optimal Price", page_icon="")
st.title("Optimal Price for Luxury Fashion Brands")
 
# Define form for user inputs
with st.form("input_form"):
    brand_name = st.selectbox("Select a Brand", ['Gucci', 'Burberry', 'Prada', 'Hermes', 'Ralph Lauren'])
    title = st.text_input("Product")
    price = st.number_input("Insert price", value=None, placeholder="Type a number...")
    cost = st.number_input("Insert production cost", value=None, placeholder="Type a number...")
    competitor_price = st.number_input("Insert competitor price", value=None, placeholder="Type a number...")
    submitted = st.form_submit_button("Submit")
# Generate random data
np.random.seed(42)

df = pd.read_csv('luxury_real_data.csv')

categorical_features = ['Brand', 'Product', 'Competitor']
numeric_features = ['Price', 'Cost', 'Competitor Price']
target_feature = 'Demand'

# Instead of LabelEncoder, try using mapping so at the end, you can reverse it
le = LabelEncoder()
df[categorical_features] = df[categorical_features].apply(
    lambda col: le.fit_transform(col))

# Define features and target

X = df.drop(target_feature, axis=1)
y = df[target_feature]

X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)

# Training model on training set
# Predicting Demands
# Training on combined training+validation set
X_train_val = pd.concat([X_train,X_val])
y_train_val = pd.concat([y_train,y_val])
rf_final = RandomForestRegressor(random_state=42)
rf_final.fit(X_train_val,y_train_val)
y_pred = rf_final.predict(X)

# Define functions for demand and profit calculation
def demand_function(item_data, model):
    return model.predict(pd.DataFrame([item_data]))[0]
 
def calculate_profit(item_data, price_range, model):
    best_price = None
    max_profit = -np.inf

    for price in price_range:
        item_data['Price'] = price
        cost = item_data['Cost']
        demand = demand_function(item_data, model)
        profit = (price - cost) * demand

        if profit > max_profit:
            max_profit = profit
            best_price = price

    return demand, best_price, max_profit

results = []

for i in range(len(X)):
    # Instance of dataset
    sample_item = X.iloc[i].copy()
    # Using predicted price as max/min price in price range
    sample_price = sample_item['Price']+100
    price_range = np.linspace(sample_item['Price'], sample_price, 100)
    # Predicting demand, price, profit
    opt_demand, opt_price, max_profit = calculate_profit(sample_item, price_range, rf_final)
    max_og = (X.iloc[i]['Price']-X.iloc[i]['Cost']) * opt_demand
    results.append([opt_demand, opt_price, max_og, max_profit])

results = pd.DataFrame(results, columns=['Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)'])
results['Product'] = df['Product']
results = results[['Product', 'Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']]

st.subheader(
    "Dataset of Optimal Demand, Prices for Maximum Profit"
)
st.dataframe(results)
