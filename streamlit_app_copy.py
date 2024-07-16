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
    title = st.selectbox("Select a Product", ['Handbag', "Women's Shoes", "Men's Shoes", "Men's Belts",
                                              "Men's Wallets", "Women's Belt", "Women's Hat", 
                                              "Women's Sunglasses", "Women's Wallet", "Men's Shirt"])
    cost = st.number_input("Insert production cost", value=None, placeholder="Type a number...")
    price = st.number_input("Insert price", value=None, placeholder="Type a number...")
    competitor_price = st.number_input("Insert competitor price", value=None, placeholder="Type a number...")
    submitted = st.form_submit_button("Submit")
# Generate random data
np.random.seed(42)

df = pd.read_csv('luxury_real_data.csv')
st.subheader('Original Dataset')
st.dataframe(df)

# Calculate stats based on input data
def calculate_stats(data):
    data['PriceDiff'] = data['Price'] - data['CompetitorPrice']
    data['Markup'] = (data['Price'] - data['Cost']) / data['Cost']
    brand_comp = {'Burberry': 'Hermes', 'Gucci':'Balenciaga', 'Prada':'Chanel', 'Versace':'Dior'}
    data['Competitor'] = data['Brand'].map(brand_comp)
    return data
# Testing
df = calculate_stats(df)

brand_map = {brand: index for index, brand in enumerate(df['Brand'].unique())}
product_map = {products: index for index, products in enumerate(df['Product'].unique())}
comp_map = {comps: index for index, comps in enumerate(df['Competitor'].unique())}

# Creating mapping for categorical values
def mappings(data):
    data['Brand'] = data['Brand'].map(brand_map)
    data['Product'] = data['Product'].map(product_map)
    data['Competitor'] = data['Competitor'].map(comp_map)
    return data

df = mappings(df)

# Define features and target
y = df['Demand']
X = df.drop(['Demand'], axis=1)
X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)

# Predicting Demands
# Training model on combined training+validation set
X_train_val = pd.concat([X_train,X_val])
y_train_val = pd.concat([y_train,y_val])
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_final.fit(X_train_val, y_train_val)

pred_demand = rf_final.predict(X)
pre_demand = sum(y)

# Predicting Price
temp = df.copy()
X_prices = pd.DataFrame(pred_demand)
y_prices = temp['Price']
X_1, X_2, y_1, y_2 = train_test_split(X_prices, y_prices, test_size=0.2, random_state=42)
model = ElasticNet(alpha=0.1,l1_ratio=0.5)
model.fit(X_1, y_1)
prices = model.predict(X_prices)

# Optimizing Profit
# Price optimization function
def optimize_demand(item_data, price_range, model):
  best_price = None
  max_profit = -np.inf

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

  return demand, best_price, max_profit

# Test the optimization for dataset
results = []

for i in range(len(X)):
  sample_item = X.iloc[i].copy()
  sample_price = prices[i]                # Using predicted price as max/min price in price range
  price_range = np.linspace(sample_item['Price'], sample_price, 100)
  opt_demand, opt_price, max_profit = optimize_demand(sample_item, price_range, rf_final)
  max_og = (X.iloc[i]['Price']-X.iloc[i]['Cost']) * opt_demand
  results.append([opt_demand, opt_price, max_og, max_profit])

results = pd.DataFrame(results, columns=['Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)'])
def reverse_stats(data):
    reverse_brandmap = {number: brand for brand, number in brand_map.items()}
    data['Brand'] = data['Brand'].map(reverse_brandmap)
    reverse_compmap = {number: comps for comps, number in comp_map.items()}
    data['Competitor'] = data['Competitor'].map(reverse_compmap)
    reverse_prodsmap = {number: prods for prods, number in product_map.items()}
    data['Product'] = data['Product'].map(reverse_prodsmap)
    return df

df = reverse_stats(df)

results['Brand'] = df['Brand']
results['Product'] = df['Product']
results['Competitor'] = df['Competitor']

results = results[['Brand', 'Product', 'Competitor', 'Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']]

st.subheader(
    "Dataset of Optimal Demand, Prices for Maximum Profit"
)
st.dataframe(results)


# Define functions for demand and profit calculation
def optimize_price(data, model):
    temp = data.copy()
    X_prices = pd.DataFrame(rf_final.predict(data.drop(['Demand'], axis=1)))
    y_prices = temp['Price']
    X_1, X_2, y_1, y_2 = train_test_split(X_prices, y_prices, test_size=0.2, random_state=42)
    temp_model = ElasticNet(alpha=0.1,l1_ratio=0.5)
    temp_model.fit(X_1, y_1)
    prices = temp_model.predict(X_prices)
    return prices

def calculate_profit(item_data, price_range, model):
    best_price = None
    max_profit = -np.inf
    
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

    return demand, best_price, max_profit


if submitted:
    # Encode user input
    data = {'Brand': brand_name, 'Product': title, 'Cost': cost,
            'Price': price, 'CompetitorPrice': competitor_price}
    data = pd.DataFrame(data, index=[0])
    data = calculate_stats(data)
    data = mappings(data)
    df = calculate_stats(df)
    df = mappings(df)
    data = data[['Brand', 'Product', 'Cost', 'Price', 'Competitor',
                'CompetitorPrice', 'PriceDiff', 'Markup']]
    data['Demand'] = rf_final.predict(data)
    # add data row into X dataframe
    data = data[['Brand', 'Product', 'Cost', 'Price', 'Competitor',
                'CompetitorPrice', 'Demand', 'PriceDiff', 'Markup']]
    df = pd.concat([df, data], ignore_index=True)

    X = df.drop('Demand', axis=1)
    temp_price = optimize_price(df, rf_final)
    results = []
    opt_price, max_profit = 0, 0
    for i in range(len(X)):
        sample_item = X.iloc[i].copy()
        sample_price = temp_price[i]                # Using predicted price as max/min price in price range
        price_range = np.linspace(sample_item['Price'], sample_price, 100)
        opt_demand, opt_price, max_profit = optimize_demand(sample_item, price_range, rf_final)
        results.append([opt_price, opt_demand, max_profit])
    st.write(f'Optimal Price: {opt_price:.2f}')
    st.write(f'Maximum Profit: {max_profit:.2f}')

    df = reverse_stats(df) # after doing predictions

    results_df = pd.DataFrame(results, columns=['Optimal Price', 'Optimal Demand', 'Max Profit (Optimal Price)'])
    full = pd.merge(df[['Brand', 'Product', 'Price', 'Demand']], results_df, left_index=True, right_index=True)
    columns = ['Brand', 'Product', 'Original Price', 'Original Demand', 'Optimal Price', 'Optimal Demand',
            'Max Profit (Optimal Price)']
    full.columns = columns

    st.subheader("Original Dataset Predictions") 
    st.table(full)

