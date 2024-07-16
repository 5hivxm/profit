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
    cost = st.number_input("Insert production cost", value=None, placeholder="Type a number...")
    price = st.number_input("Insert price", value=None, placeholder="Type a number...")
    competitor_price = st.number_input("Insert competitor price", value=None, placeholder="Type a number...")
    submitted = st.form_submit_button("Submit")
# Generate random data
np.random.seed(42)

df = pd.read_csv('luxury_real_data.csv')
# Calculate stats based on input data
def calculate_stats(data):
    data['PriceDiff'] = data['Price'] - data['CompetitorPrice']
    data['Markup'] = (data['Price'] - data['Cost']) / data['Cost']
    brand_comp = {'Burberry': 'Hermes', 'Gucci':'Balenciaga', 'Prada':'Chanel', 'Versace':'Dior'}
    data['Competitor'] = data['Brand'].map(brand_comp)
    return data
# Testing
df = calculate_stats(df)

# Creating mapping for categorical values
def mappings(data):
    brand_map = {brand: index for index, brand in enumerate(data['Brand'].unique())}
    data['Brand'] = data['Brand'].map(brand_map)
    product_map = {products: index for index, products in enumerate(data['Product'].unique())}
    data['Product'] = data['Product'].map(product_map)
    comp_map = {comps: index for index, comps in enumerate(data['Competitor'].unique())}
    data['Competitor'] = data['Competitor'].map(comp_map)
    return data

df = mappings(df)
st.dataframe(df)

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
st.write(['rf_final', rf_final.score(X_train_val, y_train_val),
                    mean_squared_error(y_test,rf_final.predict(X_test)),
                    rf_final.score(X_test, y_test)])

preds = rf_final.predict(X)
pre_demand = sum(y)

# Predicting Price
temp = df.copy()
X_prices = pd.DataFrame(preds)
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
reverse_brandmap = {number: brand for brand, number in brand_map.items()}
data['Brand'] = data['Brand'].map(reverse_brandmap)
reverse_compmap = {number: comps for comps, number in comp_map.items()}
data['Competitor'] = data['Competitor'].map(reverse_compmap)
reverse_prodsmap = {number: prods for prods, number in product_map.items()}
data['Product'] = data['Product'].map(reverse_prodsmap)
results['Brand'] = data['Brand']
results['Product'] = data['Product']
results['Competitor'] = data['Competitor']

results = results[['Brand', 'Product', 'Competitor', 'Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']]

st.subheader(
    "Dataset of Optimal Demand, Prices for Maximum Profit"
)
st.dataframe(results)


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

# Displaying Comparison Table
full = pd.merge(results, data[['Demand', 'Price', 'Profit']], left_index=True, right_index=True)
columns = ['Brand', 'Product', 'Competitor', 'Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)',
           'Max Profit (Optimal Price)', 'Original Demand', 'Original Price', 'Original Profit']
full.columns = columns
full = full[['Brand', 'Product', 'Competitor', 'Original Demand', 'Optimal Demand', 'Original Price',
             'Optimal Price', 'Original Profit', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']]
full[['Original Demand', 'Optimal Demand']] = full[['Original Demand', 'Optimal Demand']].round()
full[['Original Price', 'Optimal Price', 'Original Profit', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']] =\
full[['Original Price', 'Optimal Price', 'Original Profit', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']].round(2)

# Displaying Percent Changes and Elasticities
original_price = sum(full['Original Price'])
optimal_price = sum(full['Optimal Price'])
original_profit = sum(full['Original Profit'])
optimal_profit = sum(full['Max Profit (Optimal Price)'])
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

columns = ['Feature', 'Original Value', 'Optimized Value', 'Percent Increase', 'Elasticity on Price Change']
res = pd.DataFrame([demands, revs, profits], columns=columns)

st.subheader(
    "Full Dataset Comparing Demands, Prices, Profits"
)
st.dataframe(full)

# Plotting original vs optimal Profit
fig, ax = plt.subplots()
sns.scatterplot(x='Original Price', y='Original Profit', data=full, label='Original Prices', s=50, color='blue', ax=ax)
sns.scatterplot(x='Optimal Price', y='Max Profit (Optimal Price)', data=full, label='Optimal Prices', s=50, color='red', ax=ax)
ax.set_title('Price vs Profit')
ax.set_xlabel('Price')
ax.set_ylabel('Profit')
ax.legend()
st.pyplot(fig)

st.subheader(
    "Increases in Demand, Revenue, Profit, Elasticities"
)
st.table(res)

