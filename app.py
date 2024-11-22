from flask import Flask, request, jsonify
import pandas as pd
from datetime import timedelta, date
from geopy.geocoders import Nominatim
import time

app = Flask(__name__)

# Geolocator initialization
geolocator = Nominatim(user_agent="South Korea")

# Function to process patient data
def process_patient_data(df):
    """
    Processes a DataFrame containing patient information:
    - Calculates next visit date based on the period.
    - Computes remaining days to the next visit.
    - Adds latitude and longitude based on the address.

    Args:
        df (pd.DataFrame): DataFrame with patient data.

    Returns:
        pd.DataFrame: Processed DataFrame with added 'resDate', 'remaining_days',
                      'latitude', and 'longitude' columns.
    """
    def calculate_next_visit(row):
        """Calculates the next visit date based on the visit frequency."""
        prev_visit_date = pd.to_datetime(row['visitDate'])
        visit_freq = row['period']

        if visit_freq == 'Every 6 months':
            next_visit_date = prev_visit_date + timedelta(days=180)
        elif visit_freq == 'Every 1 year':
            next_visit_date = prev_visit_date + timedelta(days=365)
        elif visit_freq == 'Every 3 months':
            next_visit_date = prev_visit_date + timedelta(days=90)
        elif visit_freq == 'Every 2 months':
            next_visit_date = prev_visit_date + timedelta(days=60)
        else:
            next_visit_date = prev_visit_date  # Default case

        return next_visit_date.strftime('%Y-%m-%d')

    # Add 'resDate' column
    df['resDate'] = df.apply(calculate_next_visit, axis=1)
    df['resDate'] = pd.to_datetime(df['resDate'])

    # Calculate remaining days
    today = pd.to_datetime(date.today())
    df['remaining_days'] = (df['resDate'] - today).dt.days

    # Add latitude and longitude columns based on addresses
    df['latitude'] = None
    df['longitude'] = None

    for index, row in df.iterrows():
        address = row['address']
        try:
            location = geolocator.geocode(address)
            if location:
                df.at[index, 'latitude'] = location.latitude
                df.at[index, 'longitude'] = location.longitude
            else:
                df.at[index, 'latitude'] = None
                df.at[index, 'longitude'] = None
        except Exception as e:
            print(f"Error geocoding {address}: {e}")
            df.at[index, 'latitude'] = None
            df.at[index, 'longitude'] = None

        time.sleep(1)  # To avoid overloading the geocoding service

    return df

@app.route('/process', methods=['POST'])
def process_data():
    """
    Endpoint to process patient data.
    - Expects JSON data with patient information.
    - Returns processed patient data as JSON.
    """
    try:
        # Receive input data in JSON format
        input_data = request.json
        df = pd.DataFrame(input_data)

        # Process the DataFrame
        processed_df = process_patient_data(df)

        # Convert DataFrame to JSON response
        result = processed_df.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/patients', methods=['POST'])
def get_patient_list():
    """
    Endpoint to process patient data and generate a summary list.
    - Expects JSON data with patient information.
    - Returns a list of patient IDs with remaining days and locations.
    """
    try:
        # Receive input data in JSON format
        input_data = request.json
        df = pd.DataFrame(input_data)

        # Process the DataFrame
        processed_df = process_patient_data(df)

        # Generate a summary patient list
        patient_list = []
        for _, row in processed_df.iterrows():
            patient_info = {
                'patientid': row['patientid'],
                'remaining_days': row['remaining_days'],
                'location': [row['latitude'], row['longitude']]
            }
            patient_list.append(patient_info)

        return jsonify(patient_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
