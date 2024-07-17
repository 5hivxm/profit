import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Show app title and description
st.set_page_config(page_title="Optimal Price", page_icon="")
st.title("Optimal Price for Luxury Fashion Brands")
 
# Define form for user inputs
with st.form("input_form"):
    brand_name = st.selectbox("Select a Brand", ['Gucci', 'Burberry', 'Prada', 'Versace'])
    title = st.selectbox("Select a Product", ['Handbag', "Women's Shoes", "Men's Shoes", "Men's Belts",
                                              "Men's Wallets", "Women's Belt", "Women's Hat", 
                                              "Women's Sunglasses", "Women's Wallet", "Men's Shirt"])
    cost = st.number_input("Insert production cost", value=None, placeholder="Type a number...")
    price = st.number_input("Insert price", value=None, placeholder="Type a number...")
    competitor_price = st.number_input("Insert competitor price", value=None, placeholder="Type a number...")
    demand = st.number_input("Insert quantity", value=None, placeholder="Type a number...")
    submitted = st.form_submit_button("Submit")
# Generate random data
np.random.seed(42)

df = pd.read_csv('luxury_real_data.csv')
st.subheader('Original Dataset')
st.dataframe(df)

# Calculate stats based on input data
def calculate_stats(data):
    data['Profit'] = (data['Price'] - data['Cost'])*data['Demand']
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
X = df.drop(['Demand', 'Competitor'], axis=1)
X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)

correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
st.pyplot(fig)

# Random Forest Regression
# Training model on training set
rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
rf_val_error = mean_squared_error(rf.predict(X_val), y_val)
rf_test_error = mean_squared_error(rf.predict(X_test), y_test)
st.write(rf.score(X_train, y_train))
st.write(mean_squared_error(rf.predict(X_train), y_train))
st.write(rf.score(X_val, y_val))
st.write(rf_val_error)

n_estimators = [int(x) for x in np.linspace(start = 0, stop = 2000, num = 100)]
max_features = [1.0, 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 51, num = 5)]
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

print(rf_ransearch.best_params_)
rf_val_error = mean_squared_error(rf_ransearch.predict(X_val), y_val)
rf_test_error = mean_squared_error(rf_ransearch.predict(X_test), y_test)
st.write(r2_score(rf_ransearch.predict(X_test), y_test))
st.write(rf_test_error)
