# Author: Jamal Creary

import csv
import sqlite3
import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Defines car class
class Car:
    def __init__(self, ID, brand, model, year, price, color, seats, condition):
        self.ID = ID
        self.brand = brand
        self.model = model
        self.year = year
        self.price = price
        self.color = color
        self.seats = seats
        self.condition = condition

    def __str__(self):
        return f"ID: {self.ID}, Brand: {self.brand}, Model: {self.model}, Year: {self.year}, Price: ${self.price}, Color: {self.color}, Seats: {self.seats}, Condition: {self.condition}"

# Function to load car data from CSV
def load_car_data(filename):
    cars = []
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            car = Car(int(row[0]), row[1], row[2], int(row[3]), float(row[4]), row[5], int(row[6]), row[7])
            cars.append(car)
    return cars


# Load car data
cars = load_car_data('/Users/jcreary/PycharmProjects/pythonC964ProjectV8/All_Cars.csv')

# Prepare data for machine learning
car_data = {
    'brand': [car.brand for car in cars],
    'model': [car.model for car in cars],
    'year': [car.year for car in cars],
    'price': [car.price for car in cars],
    'color': [car.color for car in cars],
    'seats': [car.seats for car in cars],
    'condition': [car.condition for car in cars]
}

df = pd.DataFrame(car_data)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=[object]).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Train decision tree model
X = df[['brand', 'year', 'price', 'seats', 'condition']]
y = df['model']
model = DecisionTreeClassifier()
model.fit(X, y)


# SQLite database setup
def create_connection():
    return sqlite3.connect('car_predictions.db')

# Creates table in database
def create_table():
    conn = create_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                phone_number TEXT, 
                budget REAL,
                brand TEXT,
                condition TEXT,
                color TEXT,  
                seats INTEGER,
                vehicle_predictions TEXT  
            )
        ''')

create_table()

# Function created to save prediction results to the database
def save_result(first_name, last_name, phone_number, budget, brand, condition, color, seats, vehicle_predictions):
    conn = create_connection()
    with conn:
        conn.execute('''
            INSERT INTO predictions (first_name, last_name, phone_number, budget, brand, condition, color, seats, vehicle_predictions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (first_name, last_name, phone_number, budget, brand, condition, color, seats, vehicle_predictions))

# Function created to fetch results from the database
def fetch_results():
    conn = create_connection()
    return pd.read_sql_query('SELECT * FROM predictions', conn)

# Function creates a new window to display the results
def show_results_window(results):
    results_window = tk.Toplevel(root)
    results_window.title("Predicted Cars")

    # Create a frame for the text widget and scrollbars
    frame = tk.Frame(results_window)
    frame.pack(expand=True, fill='both')

    # Create a Text widget for displaying results
    text_widget = tk.Text(frame, wrap='word', height=20, width=80)
    text_widget.pack(side='left', fill='both', expand=True)

    # Add vertical and horizontal scrollbars
    v_scrollbar = tk.Scrollbar(frame, orient='vertical', command=text_widget.yview)
    v_scrollbar.pack(side='right', fill='y')
    text_widget.config(yscrollcommand=v_scrollbar.set)

    h_scrollbar = tk.Scrollbar(frame, orient='horizontal', command=text_widget.xview)
    h_scrollbar.pack(side='bottom', fill='x')
    text_widget.config(xscrollcommand=h_scrollbar.set)

    # Insert the results into the Text widget
    text_widget.insert('1.0', results)
    text_widget.config(state='disabled')

# Function created to calculate prediction accuracy
def calculate_prediction_accuracy(predicted_cars_count):
    total_cars_count = len(cars)
    if total_cars_count == 0:
        return "No cars available in the dataset."

    accuracy_percentage = 100 - (predicted_cars_count / total_cars_count) * 100
    return f"Prediction Accuracy: {accuracy_percentage:.2f}% based on {predicted_cars_count} out of {total_cars_count} cars in the dataset."

# Function created to predict cars based on user input
def predict_car():
    # Validate that required fields are not empty
    if not first_name_var.get() or not last_name_var.get() or not phone_number_var.get() or not budget_var.get():
        result_var.set("Please complete each field before you select 'Predict Car'.")
        return

    try:
        budget = float(budget_var.get())
    except ValueError:
        result_var.set("Invalid budget value")
        return

    try:
        seats = int(seats_var.get())
    except ValueError:
        result_var.set("Invalid seats value")
        return

    brand = brand_var.get()
    condition = condition_var.get()
    color = color_var.get()
    first_name = first_name_var.get()
    last_name = last_name_var.get()
    phone_number = phone_number_var.get()

    # Check for "Any" and handle accordingly
    if brand == "Any":
        brand_filter = df['brand'].unique()
    else:
        brand_filter = [label_encoders['brand'].transform([brand])[0]]

    if condition == "Any":
        condition_filter = df['condition'].unique()
    else:
        condition_filter = [label_encoders['condition'].transform([condition])[0]]

    # Filter cars based on the criteria
    filtered_cars = [
        car for car in cars
        if (label_encoders['brand'].transform([car.brand])[0] in brand_filter and
            label_encoders['condition'].transform([car.condition])[0] in condition_filter and
            (color == "Any" or car.color == color) and
            car.price <= budget and
            car.seats == seats)
    ]

    if not filtered_cars:
        result_var.set("No cars found that meet the criteria.")
        return

    # Prepare the result string with individual predictions
    results = [f"{car.ID}. {car.year} {car.brand} {car.model}" for car in filtered_cars]
    results_str = "\n".join(results)

    # Display results in a new window with a scrollable text widget
    show_results_window(results_str)

    # Save each result to the database as a separate row
    for car in filtered_cars:
        save_result(first_name, last_name, phone_number, budget, car.brand, car.condition, car.color,
                    car.seats, f"{car.ID}. {car.year} {car.brand} {car.model}")

    # Calculate and display prediction accuracy
    accuracy_message = calculate_prediction_accuracy(len(filtered_cars))
    result_var.set(f"{result_var.get()}\n{accuracy_message}")

# Function was created to show the three graphs of the prediction results
def show_results():
    results_df = fetch_results()

    # Plot 1: Bar Graph
    plt.figure(figsize=(10, 6), dpi=100)
    results_df['vehicle_predictions'].value_counts().plot(kind='bar')
    plt.title('Figure 1: Count of Predicted Vehicles', fontsize=14)
    plt.xlabel('Predicted Vehicle', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=90, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.35)
    plt.show()


    # Plot 2: Scatter Plot with Regression Line
    # Extract budgets and seats
    budgets = results_df['budget']
    seats = results_df['seats']

    plt.figure(figsize=(10, 6), dpi=100)

    # Scatter plot
    plt.scatter(budgets, seats, alpha=0.6)
    plt.title('Figure 2: Budget vs Seats', fontsize=14)
    plt.xlabel('Budget', fontsize=12)
    plt.ylabel('Seats', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Setting the y-axis range from 0 to 7 for a better visual even though the seats range from 2-7
    plt.ylim(0, 7)

    # Adding the regression line
    X = budgets.values.reshape(-1, 1)
    y = seats.values
    reg = LinearRegression().fit(X, y)
    plt.plot(budgets, reg.predict(X), color='red', linewidth=2)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.grid(True)
    plt.show()

    # Plot 3: Count of Predictions by Budget Range (Line Graph)
    plt.figure(figsize=(10, 6), dpi=100)
    budget_max = int(results_df['budget'].max())
    budget_ranges = pd.cut(results_df['budget'], bins=range(0, budget_max + 10000, 10000))
    count_by_range = budget_ranges.value_counts().sort_index()
    count_by_range.plot(kind='line', marker='o')
    plt.title('Figure 3: Count of Predictions by Budget Range', fontsize=14)
    plt.xlabel('Budget Range', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    plt.grid(True)
    plt.show()

# Function created to view database
def view_database():
    results_df = fetch_results()
    view_window = tk.Toplevel(root)
    view_window.title("Database Contents")

    # Create a frame to hold the Treeview and scrollbars
    frame = tk.Frame(view_window)
    frame.pack(expand=True, fill='both')

    # Create a Treeview widget
    tree = ttk.Treeview(frame, columns=list(results_df.columns), show='headings')
    tree.grid(row=0, column=0, sticky='nsew')

    # Add vertical scrollbar
    vsb = tk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    tree.configure(yscrollcommand=vsb.set)

    # Add horizontal scrollbar
    hsb = tk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    hsb.grid(row=1, column=0, sticky='ew')
    tree.configure(xscrollcommand=hsb.set)

    # Define the columns
    for col in results_df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    # Insert the data into the Treeview
    for _, row in results_df.iterrows():
        tree.insert('', 'end', values=list(row))

    # Adjust column widths to fit content
    for col in results_df.columns:
        tree.column(col, width=max(tree.column(col, 'width'), 150))

    # Make sure the grid expands to fill the window
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

# Function created to view the most predicted car brand
def most_predicted_brand():
    results_df = fetch_results()

    # Ensure 'vehicle_predictions' is not empty
    if results_df.empty or 'vehicle_predictions' not in results_df.columns:
        result_var.set("No predictions available.")
        return

    # Extract and clean predictions
    predictions = results_df['vehicle_predictions'].dropna().str.split(", ", expand=True).stack().reset_index(drop=True)

    # Extract ID from the prediction string
    def extract_id(prediction):
        try:
            return prediction.split('.')[0].strip()
        except IndexError:
            return None

    # Create a DataFrame from the predictions
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])
    predictions_df['ID'] = predictions_df['prediction'].apply(extract_id)

    # Load the original car data for brand lookup
    car_df = pd.DataFrame({
        'ID': [str(car.ID) for car in cars],
        'brand': [car.brand for car in cars]
    })

    # Merge the predictions with the car DataFrame to get brands
    merged_df = predictions_df.merge(car_df, on='ID', how='left')

    # Check if the merge was successful
    if merged_df['brand'].isnull().all():
        result_var.set("No matching brands found.")
        return

    # Find the most common brands
    brand_counts = merged_df['brand'].dropna().value_counts()
    max_count = brand_counts.max()
    most_common_brands = brand_counts[brand_counts == max_count].index.tolist()

    # Display the results
    if most_common_brands:
        result_var.set(f"The most recommended brand(s) is:\n{', '.join(most_common_brands)}")
    else:
        result_var.set("No predictions available.")

# Function created to view all car data in a new window
def view_all_cars():
    # Reload car data from CSV
    cars = load_car_data('/Users/jcreary/PycharmProjects/pythonC964ProjectV8/All_Cars.csv')

    view_window = tk.Toplevel(root)
    view_window.title("All Cars on Lot")

    # Create a Treeview widget
    tree = ttk.Treeview(view_window, columns=["ID", "Brand", "Model", "Year", "Price", "Color", "Seats", "Condition"],
                        show='headings')
    tree.pack(expand=True, fill='both')

    # Define the columns
    for col in ["ID", "Brand", "Model", "Year", "Price", "Color", "Seats", "Condition"]:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Clear existing items in the Treeview
    for item in tree.get_children():
        tree.delete(item)

    # Insert the updated data into the Treeview
    for car in cars:
        tree.insert('', 'end',
                    values=(car.ID, car.brand, car.model, car.year, car.price, car.color, car.seats, car.condition))

# Create main window
root = tk.Tk()
root.title("Car Predictor")

# Create frame for user preferences
pref_frame = tk.Frame(root)
pref_frame.pack(pady=10)

# Define Tkinter StringVars before using them
first_name_var = tk.StringVar()
last_name_var = tk.StringVar()
phone_number_var = tk.StringVar()
budget_var = tk.StringVar()
brand_var = tk.StringVar(value="Any")
condition_var = tk.StringVar(value="Any")
color_var = tk.StringVar(value="Any")
seats_var = tk.StringVar(value="Select a value")

# Set uniform width for entry and combobox widgets
uniform_width = 22
row_padding = 2

# Add labels and entries to the grid
labels = [
    "First Name:", "Last Name:", "Phone Number:", "Budget ($):",
    "Brand:", "Condition:", "Color:", "Seats:"
]
variables = [
    first_name_var, last_name_var, phone_number_var, budget_var,
    brand_var, condition_var, color_var, seats_var
]

for i, (label, var) in enumerate(zip(labels, variables)):
    tk.Label(pref_frame, text=label, anchor='w').grid(row=i, column=0, padx=5, pady=row_padding, sticky='w')

    if label == "Brand:":
        brands = sorted(list(set(car.brand for car in cars)))
        entry = ttk.Combobox(pref_frame, textvariable=var, width=uniform_width - 1, state='readonly')
        entry['values'] = ["Any"] + brands
    elif label == "Condition:":
        entry = ttk.Combobox(pref_frame, textvariable=var, width=uniform_width - 1, state='readonly')
        entry['values'] = ["Any"] + list(set(car.condition for car in cars))
    elif label == "Color:":
        entry = ttk.Combobox(pref_frame, textvariable=var, width=uniform_width - 1, state='readonly')
        entry['values'] = ["Any"] + list(set(car.color for car in cars))
    elif label == "Seats:":
        entry = ttk.Combobox(pref_frame, textvariable=var, width=uniform_width - 1, state='readonly')
        entry['values'] = list(range(2, 8))
    else:
        entry = tk.Entry(pref_frame, textvariable=var, width=uniform_width)

    entry.grid(row=i, column=1, padx=5, pady=row_padding, sticky="w")

# Add buttons to a separate frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Predict button
predict_button = tk.Button(button_frame, text="Predict Car", command=predict_car)
predict_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

# Show results button
graph_button = tk.Button(button_frame, text="Show Graphs", command=show_results)
graph_button.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

# View all cars on lot button
view_all_cars_button = tk.Button(button_frame, text="View All Cars", command=view_all_cars)
view_all_cars_button.grid(row=3, column=0, padx=5, pady=5, sticky='ew')

# View database button
database_button = tk.Button(button_frame, text="View Input Database", command=view_database)
database_button.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

# Most predicted brand button
most_predicted_button = tk.Button(button_frame, text="Most Common Predicted Brand(s)", command=most_predicted_brand)
most_predicted_button.grid(row=4, column=0, padx=5, pady=5, sticky='ew')

# Set uniform column width
button_frame.columnconfigure(0, weight=1)

# Result display
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, justify='left')
result_label.pack(pady=5)

# Starts the event loop
root.mainloop()
