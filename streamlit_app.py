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
X = df.drop(['Demand'], axis=1)
X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)

# Random Forest Regression
# Training model on training set
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_val_error = mean_squared_error(rf.predict(X_val), y_val)
rf_test_error = mean_squared_error(rf.predict(X_test), y_test)
st.write(rf.score(X_train, y_train))
st.write(mean_squared_error(rf.predict(X_train), y_train))
st.write(rf.score(X_val, y_val))
st.write(rf_val_error)

pred_demand = rf.predict(X)
pre_profit = sum(df['Profit'])
pre_demand = sum(y)

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


# Predicting Price
def optimize_price(data, model):
    temp = data.copy()
    X_prices = pd.DataFrame(model.predict(data.drop(['Demand'], axis=1)))
    y_prices = temp['Price']
    X_1, X_2, y_1, y_2 = train_test_split(X_prices, y_prices, test_size=0.2, random_state=42)
    temp_model = ElasticNet(alpha=0.1,l1_ratio=0.5)
    temp_model.fit(X_1, y_1)
    prices = temp_model.predict(X_prices)
    return prices

prices = optimize_price(df, rf_final)

# Optimizing Profit
# Price optimization function
def calculate_profit(item_data, price_range, model, og_demand):
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

    return best_demand, best_price, max_profit
# Test the optimization for dataset
results = []

for i in range(len(X)):
    sample_item = X.iloc[i].copy()
    og_demand = y.iloc[i]
    sample_price = prices[i]                # Using predicted price as max/min price in price range
    price_range = np.linspace(sample_item['Price'], sample_price, 100)
    opt_demand, opt_price, max_profit = calculate_profit(sample_item, price_range, rf_final, og_demand)
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
results = results[['Brand', 'Product', 'Competitor', 'Optimal Demand', 
                   'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']]
st.subheader(
    "Dataset of Optimal Demand, Prices for Maximum Profit"
)
st.dataframe(results)


if submitted:
    # Encode user input
    data = {'Brand': brand_name, 'Product': title, 'Cost': cost,
            'Price': price, 'CompetitorPrice': competitor_price, 'Demand': demand}
    data = pd.DataFrame(data, index=[0])
    data = calculate_stats(data)
    data = mappings(data)
    df = calculate_stats(df)
    df = mappings(df)
    data = data[['Brand', 'Product', 'Cost', 'Price', 'Competitor',
                'CompetitorPrice', 'Demand', 'Profit', 'PriceDiff', 'Markup']]
    # add data row into X dataframe
    df = pd.concat([df, data], ignore_index=True)

    X = df.drop(['Demand'], axis=1)
    y = df['Demand']
    temp_price = optimize_price(df, rf_final)
    results = []
    opt_price, max_profit = 0, 0
    for i in range(len(X)):
        sample_item = X.iloc[i].copy()
        og_demand = y.iloc[i]
        sample_price = temp_price[i]                # Using predicted price as max/min price in price range
        price_range = np.linspace(sample_item['Price'], sample_price, 100)
        opt_demand, opt_price, max_profit = calculate_profit(sample_item, price_range, rf_final, og_demand)
        max_og = (X.iloc[i]['Price']-X.iloc[i]['Cost']) * opt_demand
        results.append([opt_price, opt_demand, max_og, max_profit])

    df = reverse_stats(df) # after doing predictions

    results_df = pd.DataFrame(results, columns=['Optimal Price', 'Optimal Demand',
                                                'Max Profit (Original Price)', 'Max Profit (Optimal Price)'])

    full = pd.merge(df[['Brand', 'Product', 'Price', 'Demand', 'Profit']], results_df, left_index=True, right_index=True)
    columns = ['Brand', 'Product', 'Original Price', 'Original Demand', 'Original Profit', 
               'Optimal Price', 'Optimal Demand', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']
    full.columns = columns

    st.subheader("Original Dataset Predictions") 
    st.dataframe(full)
    st.write(f'Optimal Price: {opt_price:.2f}')
    st.write(f'Maximum Profit: {max_profit:.2f}')

    original_price = sum(full['Original Price'])
    optimal_price = sum(full['Optimal Price'])
    original_profit = sum(full['Original Profit'])
    optimal_profit = sum(full['Max Profit (Optimal Price)'])
    original_demand = sum(full['Original Demand'])
    optimal_demand = sum(full['Optimal Demand'])
    original_revenue = sum(full['Original Price']*full['Original Demand'])
    optimal_revenue = sum(full['Optimal Price']*full['Optimal Demand'])

    demands = ['Demand', original_demand, optimal_demand, (optimal_demand-original_demand)/original_demand*100,
            (optimal_demand-original_demand)/(optimal_price - original_price)]
    revs = ['Revenue', original_revenue, optimal_revenue, (optimal_revenue - original_revenue)/original_revenue*100,
            (optimal_revenue-original_revenue)/(optimal_price - original_price)]
    profits = ['Profit', original_profit, optimal_profit, (optimal_profit-original_profit)/original_profit*100,
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

