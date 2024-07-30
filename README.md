# Vehicle-Prediction-Application
Vehicle Prediction Model

Overview

This project is a Car Prediction Application that provides users with recommendations for cars based on their preferences. The application uses a machine learning model to predict cars and integrates a SQLite database to store user inputs and predictions.

Features

Load Car Data: Load car data from a CSV file.
Car Prediction: Predict cars based on user inputs such as brand, budget, condition, color, and seats.
Save Predictions: Save prediction results to a SQLite database.
View Predictions: Fetch and view prediction results from the database.
Graphical Analysis: Display graphs to analyze prediction results.
View All Cars: View all available cars in the dataset.
Most Predicted Brand: Find and display the most recommended car brands.
Dependencies

csv: To handle CSV file operations.
sqlite3: To interact with the SQLite database.
tkinter: For creating the graphical user interface.
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib: For creating plots and visualizations.
sklearn: For machine learning models and preprocessing.
Installation

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/car-prediction-app.git
cd car-prediction-app
Install the required packages:
bash
Copy code
pip install pandas numpy matplotlib scikit-learn
Run the application:
bash
Copy code
python car_prediction_app.py
Usage

Load Car Data: Ensure the car data CSV file is located at /Users/jcreary/PycharmProjects/pythonC964ProjectV8/All_Cars.csv.
User Input: Fill in the user preferences for first name, last name, phone number, budget, brand, condition, color, and seats.
Predict Car: Click the "Predict Car" button to get car recommendations based on user inputs.
Show Graphs: Click the "Show Graphs" button to display graphical analysis of prediction results.
View All Cars: Click the "View All Cars" button to see all available cars in the dataset.
View Input Database: Click the "View Input Database" button to view the stored prediction results.
Most Common Predicted Brand(s): Click the "Most Common Predicted Brand(s)" button to see the most frequently recommended car brands.
File Structure

car_prediction_app.py: Main application file containing the code for the car prediction application.
All_Cars.csv: CSV file containing the car data.
README.md: This readme file.
