from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import mysql.connector
from mysql.connector import Error
import logging
import config

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection details
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Aditya.6404'
DB_NAME = 'regression_investment_db'

# Establish a connection to the database
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

# Create the required table if it doesn't exist
def create_table_if_not_exists():
    connection = get_db_connection()
    if connection:
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            `Depreciation and amortization` FLOAT,
            EBITDA FLOAT,
            Inventory FLOAT,
            `Net Income` FLOAT,
            `Total Receivables` FLOAT,
            `Market value` FLOAT,
            `Total assets` FLOAT,
            `Total Current Liabilities` FLOAT,
            `Total Long-term Debt` FLOAT,
            `Total Revenue` FLOAT,
            prediction FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        connection.close()

# Ensure the table exists before starting
create_table_if_not_exists()

# Define preprocessing function
def preprocess_features(data):
    original_features = {key: float(data.get(key, 0)) for key in [
        'Depreciation and amortization', 'EBITDA', 'Inventory', 'Net Income',
        'Total Receivables', 'Market value', 'Total assets', 'Total Current Liabilities',
        'Total Long-term Debt', 'Total Revenue'
    ]}

    processed_features = [
        original_features['Depreciation and amortization'],
        original_features['EBITDA'],
        original_features['Inventory'],
        original_features['Net Income'],
        original_features['Total Receivables'],
        original_features['Market value'],
        original_features['Total assets'],
        original_features['EBITDA'] / original_features['Total Revenue'] if original_features['Total Revenue'] != 0 else 0,
        original_features['Net Income'] / original_features['Total Revenue'] if original_features['Total Revenue'] != 0 else 0,
        (original_features['Total Current Liabilities'] + original_features['Total Long-term Debt']) / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        original_features['Net Income'] / (original_features['Total assets'] + original_features['Total Current Liabilities'] + original_features['Total Long-term Debt']) if (original_features['Total assets'] + original_features['Total Current Liabilities'] + original_features['Total Long-term Debt']) != 0 else 0,
        original_features['Net Income'] / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        original_features['Total Revenue'] / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        (original_features['Total assets'] - original_features['Total Current Liabilities']) / original_features['Total Revenue'] if original_features['Total Revenue'] != 0 else 0,
        original_features['Total Long-term Debt'] / (original_features['Total assets'] - original_features['Total Current Liabilities']) if (original_features['Total assets'] - original_features['Total Current Liabilities']) != 0 else 0,
        original_features['Market value'] / original_features['Net Income'] if original_features['Net Income'] != 0 else 0,
    ]

    scaled_features = scaler.transform([processed_features])
    return scaled_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_investment():
    try:
        data = request.json
        logger.info(f"Received input data: {data}")

        required_features = [
            'Depreciation and amortization', 'EBITDA', 'Inventory', 'Net Income',
            'Total Receivables', 'Market value', 'Total assets', 'Total Current Liabilities',
            'Total Long-term Debt', 'Total Revenue'
        ]

        for feature in required_features:
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
            try:
                float(data[feature])
            except ValueError:
                raise ValueError(f"Invalid input: {feature} must be a numeric value.")

        processed_features = preprocess_features(data)
        logger.info(f"Processed features: {processed_features}")

        prediction = model.predict(processed_features)[0]
        logger.info(f"Prediction result: {prediction}")

        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            insert_query = """
            INSERT INTO predictions (
                `Depreciation and amortization`, EBITDA, Inventory, `Net Income`,
                `Total Receivables`, `Market value`, `Total assets`, `Total Current Liabilities`,
                `Total Long-term Debt`, `Total Revenue`, prediction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            cursor.execute(insert_query, (
                data['Depreciation and amortization'], data['EBITDA'], data['Inventory'],
                data['Net Income'], data['Total Receivables'], data['Market value'],
                data['Total assets'], data['Total Current Liabilities'], data['Total Long-term Debt'],
                data['Total Revenue'], prediction
            ))
            connection.commit()
            cursor.close()
            connection.close()

        return jsonify({'prediction': float(prediction)})
    
    except ValueError as ve:
        logger.error(f"Validation Error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = config.PORT_NUMBER, debug=False)
