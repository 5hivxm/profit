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

mapping = {index: value for index, value in df['Brand'].index}
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

def optimize_price(original, predicted, model, df, X):
    temp = df.copy()
    X_prices = pd.DataFrame(model.predict(X))
    y_prices = temp['Price']
    X_1, X_2, y_1, y_2 = train_test_split(X_prices, y_prices, test_size=0.2, random_state=42)

    model = ElasticNet(alpha=0.1,l1_ratio=0.5)
    model.fit(X_1, y_1)
    prices = model.predict(X)
    return prices

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

if submitted:
    # Encode user input
    data = {'Brand': brand_name, 'Product': title, 'Price': price,
            'Cost': cost, 'Competitor Price': competitor_price}
    data = pd.DataFrame(data)
    # Insert mapping for competitor to match the brand
    data['Competitor'] = 1

    categorical_features = ['Brand', 'Product', 'Competitor']
    numeric_features = ['Price', 'Cost', 'Competitor Price']
    target_feature = 'Demand'

    data[categorical_features] = data[categorical_features].apply(
        lambda col: le.fit_transform(col))

    # add data row into X dataframe
    tempY = model.predict(data)
   
    # Predict demand for user input
    user_predicted_demand = model.predict(data)[0]
 
    # Optimize price for user input
    optimal_price, max_profit = optimize_price(cost, competitor_price, user_predicted_demand)
   
    # Display results
    st.write(f'Optimal Price: {optimal_price:.2f}')
    st.write(f'Maximum Profit: {max_profit:.2f}')


#results = []

#for idx, row in X_test.iterrows():
#    production_cost = row['ProductionCost']
#    competitor_price = row['CompetitorPrice']
#    predicted_demand = y_pred[idx]
#   
#    optimal_price, max_profit = optimize_price(production_cost, competitor_price, predicted_demand)
#   
#    results.append({
#        'BrandName': row['BrandName'],
#        'ItemNumber': row['ItemNumber'],
#        'ProductionCost': production_cost,
#        'CompetitorPrice': competitor_price,
#        'PredictedDemand': predicted_demand,
#        'OptimizedPrice': optimal_price
#    })
 
# Format results
#results_df = pd.DataFrame(results)
#st.subheader("Original Dataset Predictions") 
#st.table(results_df)
