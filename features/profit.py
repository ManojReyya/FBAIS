from flask import Flask, render_template, request, jsonify
from prediction_pipeline import ProfitabilityPredictor
import traceback
import pandas as pd
import json
import numpy as np

app = Flask(__name__)
predictor = ProfitabilityPredictor()

df = pd.read_csv('final_dataset.csv')
with open('cities_data.json', 'r') as f:
    cities_data = json.load(f)
with open('establishments_data.json', 'r') as f:
    establishments_data = json.load(f)
with open('cuisines_data.json', 'r') as f:
    cuisines_data = json.load(f)

@app.route('/')
def index():
    return render_template('profit.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        user_input = {
            'city': data.get('city'),
            'establishment_type': data.get('establishment_type'),
            'cuisines': [c.strip() for c in data.get('cuisines', '').split(',') if c.strip()],
            'opening_hours': float(data.get('opening_hours', 10)),
            'latitude': float(data.get('latitude')),
            'longitude': float(data.get('longitude')),
            'average_cost_for_two': int(data.get('average_cost_for_two', 500)),
            'aggregate_rating': float(data.get('aggregate_rating', 3.5)),
            'votes': int(data.get('votes', 100)),
            'total_cuisines': int(data.get('total_cuisines', 1)),
            'nearest_city_population': int(data.get('nearest_city_population', 1000000))
        }
        
        result = predictor.predict(user_input)
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/cities', methods=['GET'])
def get_cities():
    cities = sorted(list(cities_data.keys()))
    return jsonify({
        'cities': cities,
        'count': len(cities)
    })

@app.route('/establishments', methods=['GET'])
def get_establishments():
    return jsonify({
        'establishments': establishments_data['establishments'],
        'count': establishments_data['count']
    })

@app.route('/cuisines', methods=['GET'])
def get_cuisines():
    return jsonify({
        'cuisines': cuisines_data['cuisines'],
        'count': cuisines_data['count']
    })

@app.route('/city-info/<city>', methods=['GET'])
def get_city_info(city):
    if city not in cities_data:
        return jsonify({'error': 'City not found'}), 404
    
    info = cities_data[city]
    return jsonify(info)

@app.route('/location-stats', methods=['POST'])
def get_location_stats():
    try:
        data = request.get_json()
        city = data.get('city')
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        
        if city not in df['city'].values:
            return jsonify({'error': 'City not found'}), 400
        
        city_df = df[df['city'] == city]
        
        distances = np.sqrt((city_df['latitude'] - latitude)**2 + (city_df['longitude'] - longitude)**2)
        nearby_restaurants = city_df[distances < 0.05]
        
        if len(nearby_restaurants) == 0:
            avg_rating = city_df['aggregate_rating'].mean()
            avg_votes = city_df['votes'].mean()
        else:
            avg_rating = nearby_restaurants['aggregate_rating'].mean()
            avg_votes = nearby_restaurants['votes'].mean()
        
        city_population = city_df['nearest_city_population'].iloc[0]
        
        return jsonify({
            'success': True,
            'nearby_count': len(nearby_restaurants),
            'avg_rating': round(float(avg_rating), 2),
            'avg_votes': int(avg_votes),
            'city_population': int(city_population),
            'avg_cost': round(float(city_df['average_cost_for_two'].mean()), 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': '93-95% Accuracy (Random Forest)',
        'version': '1.0',
        'cities': len(cities_data),
        'restaurants': len(df)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
