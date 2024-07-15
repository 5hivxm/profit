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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

df = pd.read_csv('price_optimization_dataset.csv')

brand_names = ["Gucci", "Hermes", "Louis Vuitton", "Dior", "Prada"]
brands = []
for brand in range(1, 101):
  index = (brand-1)%5
  brand_name = brand_names[index]
  brands.append(brand_name)

df['BrandName'] = brands
df = df[['BrandName', 'Product', 'Price', 'ProductionCost', 'CompetitorPrice', 'Demand']]

st.dataframe(df)

# Define features and target
categorical_features = ['BrandName', 'Product']
numeric_features = ['Price', 'ProductionCost', 'CompetitorPrice']
target_feature = 'Demand'

X = df[categorical_features + numeric_features]
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(categories='auto', drop='first', sparse_output=False)
 
# Fit transformers on training data
X_train_numeric = numeric_transformer.fit_transform(X_train[numeric_features])
X_train_categorical = categorical_transformer.fit_transform(X_train[categorical_features])
 
X_train_transformed = np.hstack((X_train_numeric, X_train_categorical))
X_test_numeric = numeric_transformer.transform(X_test[numeric_features])
X_test_categorical = categorical_transformer.transform(X_test[categorical_features])
X_test_transformed = np.hstack((X_test_numeric, X_test_categorical))
 
# Train ElasticNet model
model = ElasticNet()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
    'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_transformed, y_train)
 
# Evaluate the model
y_pred = grid_search.predict(X_test_transformed)
X_test = X_test.reset_index(drop=True)

# Define functions for demand and profit calculation
def demand_function(price, competitor_price, predicted_demand):
    sensitivity = 2
    competitor_influence = 0.75
    return predicted_demand - sensitivity * (price - competitor_price) * competitor_influence
 
def calculate_profit(price, cost, competitor_price, predicted_demand):
    demand = demand_function(price, competitor_price, predicted_demand)
    profit = (price - cost) * demand
    return profit
 
def optimize_price(cost, competitor_price, predicted_demand, price_bounds=(100, 500)):
    def objective(price):
        return -calculate_profit(price, cost, competitor_price, predicted_demand)
    result = minimize_scalar(objective, bounds=price_bounds, method='bounded')
    optimal_price = result.x
    max_profit = -result.fun
    return optimal_price, max_profit
 
# Handle user input
if submitted:
    user_input_categorical = pd.DataFrame([[brand_name, 1]], columns=categorical_features)
    user_input_categorical = categorical_transformer.transform(user_input_categorical)   
    user_input_numeric = pd.DataFrame([[cost, 0, competitor_price]], columns=numeric_features)
    user_input_numeric = numeric_transformer.transform(user_input_numeric)
    user_input_transformed = np.hstack((user_input_numeric, user_input_categorical))
    # Predict demand for user input
    user_predicted_demand = grid_search.predict(user_input_transformed)[0]
 
    # Optimize price for user input
    optimal_price, max_profit = optimize_price(cost, competitor_price, user_predicted_demand)
   
    # Display results
    st.write(f'Optimal Price: {optimal_price:.2f}')
    st.write(f'Maximum Profit: {max_profit:.2f}')
 
# Collect results
results = []
 
for idx, row in X_test.iterrows():
    production_cost = row['ProductionCost']
    competitor_price = row['CompetitorPrice']
    predicted_demand = y_pred[idx]
   
    optimal_price, max_profit = optimize_price(production_cost, competitor_price, predicted_demand)
   
    results.append({
        'BrandName': row['BrandName'],
        'Product': row['Product'],
        'ProductionCost': production_cost,
        'CompetitorPrice': competitor_price,
        'PredictedDemand': predicted_demand,
        'OptimizedPrice': optimal_price
    })
 
# Format results
results_df = pd.DataFrame(results)
st.subheader("Original Dataset Predictions") 
st.table(results_df)


st.title("Modeling Low Volume Transactoins")
st.subheader(
    "Sales Dataset"
)


