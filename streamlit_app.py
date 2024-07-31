import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

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

# Main Streamlit app code
@st.cache_data
def main():

    filename = "random_forest.pickle"
    rf = pickle.load(open(filename, "rb"))
    results = pd.read_csv('results.csv')
    df = pd.read_csv('df.csv')

    with open('brand_map.json', 'r') as f: 
        brand_map = json.load(f)

    with open('product_map.json', 'r') as f: 
        product_map = json.load(f)

    return rf, df, brand_map, product_map, results

# User input fields
def plots(brand_data, brand_name, brand_change, title, item_data, item_change):
    # Display percent increases for item
    st.write(f"Optimal Price: {round(item_data.loc[len(item_data)-1, 'Optimal Price'], 2)}")
    st.write(f"Predicted Demand: {int(item_data.loc[len(item_data)-1, 'Optimal Demand'])}")

    # Display percent increases for item
    st.subheader(f"Increases in Demand, Revenue, Profit for {brand_name}'s {title}")
    features = item_change['Feature']
    cols = item_change.columns[1:]
    temp = pd.DataFrame({col: item_change[col].apply(lambda x: f"{x:.2f}") for col in cols})
    temp = pd.concat([features, temp], axis=1)
    st.table(temp)
    st.divider()

    # Display all company's product results
    st.header(f"Optimized Demand, Price, Profits for {brand_name}")
    features = brand_data.iloc[:, 1]
    cols = brand_data.columns[2:]
    temp = pd.DataFrame({col: brand_data[col].apply(lambda x: f"{x:.2f}") for col in cols})
    temp = pd.concat([features, temp], axis=1)
    st.table(temp)

    # Plot for all of company's products
    fig, ax = plt.subplots()
    ind = brand_data.index
    width = 0.25
    xvals = brand_data['Original Profit']
    bar1 = plt.bar(ind, xvals, width, color = 'r') 
    yvals = brand_data['Max Profit (Original Price)']
    bar2 = plt.bar(ind+width, yvals, width, color='g') 
    zvals = brand_data['Max Profit (Optimal Price)']
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'b') 
    plt.xlabel("Product")
    plt.ylabel("Profit")
    plt.title(f"Increases in Profits for {brand_name} Products")
    plt.xticks(ind+width, brand_data['Product']) 
    plt.legend( (bar1, bar2, bar3), ('Original Profit', 'Max Profit (Original Price, Optimal Demand)',
                                     'Max Profit (Optimal Price, Optimal Demand)') ) 
    plt.xticks(rotation=70)
    st.pyplot(fig)

    st.divider()
    # Display percent increases for company
    st.subheader(f"Increases in Demand, Revenue, Profit for {brand_name} Company")
    features = brand_change['Feature']
    cols = brand_change.columns[1:]
    temp = pd.DataFrame({col: brand_change[col].apply(lambda x: f"{x:.2f}") for col in cols})
    temp = pd.concat([features, temp], axis=1)
    st.table(temp)


def buttons():
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

    if submitted:
        # Encode user input
        rf, df, brand_map, product_map, results = main()

        data = {'Brand': brand_name, 'Product': title, 'Cost': cost,
                'Price': price, 'CompetitorPrice': competitor_price, 'Demand': demand}
        data = pd.DataFrame(data, index=[0])
        data = calculate_stats(data)
        data = mappings(data, brand_map, product_map)
        data = data[['Brand', 'Product', 'Cost', 'Price',
                    'CompetitorPrice', 'Demand', 'Profit', 'PriceDiff', 'Markup']]
        X = data.drop(['Demand'], axis=1)
        y = data['Demand'] 
        new_row=[]
        graph_data={}

        sample_item = X.iloc[0].copy()
        og_demand = y.iloc[0]
        sample_price = sample_item['Price'] + 100
        price_range = np.linspace(sample_item['Price'], sample_price, 50)
        opt_demand, opt_price, max_profit, graph_data = calculate_profit(sample_item, price_range, rf, og_demand, graph_data)
        max_og = (X.iloc[0]['Price']-X.iloc[0]['Cost']) * opt_demand
        
        data = reverse_stats(data, brand_map, product_map) # after doing predictions    
        new_row.append([opt_demand, opt_price, max_og, max_profit])
        new_row = pd.DataFrame(new_row, columns=['Optimal Demand', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)'])
        new_row['Brand'] = data['Brand']
        new_row['Product'] = data['Product']
        new_row['Original Price'] = data['Price']
        new_row['Original Demand'] = data['Demand']
        new_row['Original Profit'] = data['Profit']
        new_row = new_row[['Brand', 'Product', 'Original Price', 'Original Demand', 'Original Profit', 'Optimal Price', 
                        'Optimal Demand', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']]
        results = pd.concat([results, new_row], ignore_index=True)

        df = pd.concat([df, data], ignore_index=True)
        brand_data = results[results['Brand'] == brand_name].reset_index().drop(columns=['index'])
        demands, revs, profits = changes(brand_data)
        brand_change = pd.DataFrame([demands, revs, profits], columns=['Feature', 'Original Value', 'Optimized Value', 'Percent Increase'])

        item_data = new_row.reset_index()
        demands, revs, profits = changes(item_data)
        item_change = pd.DataFrame([demands, revs, profits], columns=['Feature', 'Original Value', 'Optimized Value', 'Percent Increase'])

        # Display data for selected brand
        brand_data[['Original Demand', 'Optimal Demand']] = brand_data[['Original Demand', 'Optimal Demand']].round()
        brand_data[['Original Price', 'Original Profit', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']] =\
            brand_data[['Original Price', 'Original Profit', 'Optimal Price', 'Max Profit (Original Price)', 'Max Profit (Optimal Price)']].round(2)

        plots(brand_data, brand_name, brand_change, title, item_data, item_change)



if __name__ == '__main__':
    st.title("Optimal Price for Luxury Fashion Brands")
    main()
    buttons()