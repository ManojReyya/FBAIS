from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
import os
import math
from pytrends.request import TrendReq
import requests
from collections import defaultdict
import holidays
import logging

logging.basicConfig(level=logging.WARNING)

app = Flask(__name__)

FESTIVAL_CALENDAR = {
    1: ['New Year', 'Republic Day'],
    2: ['Valentine Day', 'Presidents Day'],
    3: ['Holi', 'St. Patricks Day'],
    4: ['Easter'],
    5: ['Ramzan', 'Summer Start'],
    6: ['Father Day'],
    7: [],
    8: ['Independence Day', 'Raksha Bandhan'],
    9: ['Janmashtami', 'Labor Day'],
    10: ['Dussehra', 'Diwali Prep', 'Halloween'],
    11: ['Diwali', 'Thanksgiving'],
    12: ['Christmas', 'New Year Prep']
}

SEASONAL_FOODS = {
    'winter': ['hot_beverages', 'soups', 'warming_spices', 'meat_dishes', 'comfort_food'],
    'summer': ['cold_drinks', 'salads', 'ice_cream', 'light_meals', 'fruits'],
    'monsoon': ['hot_tea', 'fritters', 'pakoras', 'warming_foods'],
    'spring': ['fresh_vegetables', 'salads', 'light_foods', 'seasonal_fruits']
}

LOCAL_TASTE_PATTERNS = {
    'North India': ['butter_chicken', 'tandoori', 'dal_makhni', 'samosa', 'lassi'],
    'South India': ['dosa', 'idli', 'sambar', 'coconut', 'rice'],
    'East India': ['fish', 'rice', 'mustard', 'sweets'],
    'West India': ['snacks', 'street_food', 'dal_dhokli', 'falafel'],
    'North East India': ['bamboo_shoots', 'fish', 'rice', 'momos', 'local_curry'],
    'Central India': ['wheat', 'lentils', 'jaggery', 'local_vegetables', 'dal']
}

REGION_COUNTRIES = {
    'North India': 'IN',
    'South India': 'IN',
    'East India': 'IN',
    'West India': 'IN',
    'North East India': 'IN',
    'Central India': 'IN'
}

WEATHER_DEMAND_MAP = {
    'hot': ['ice_cream', 'cold_drinks', 'salads', 'light_meals', 'smoothies'],
    'cold': ['hot_beverages', 'soups', 'warm_meals', 'meat_dishes', 'comfort_food'],
    'rainy': ['hot_tea', 'snacks', 'warming_foods', 'pakoras', 'bread_items'],
    'clear': ['light_meals', 'salads', 'fresh_juices', 'outdoor_food']
}

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8, 9]:
        return 'monsoon'
    else:
        return 'summer'

def get_weather_condition(temp, humidity, weather_type):
    if temp > 30:
        return 'hot'
    elif temp < 10:
        return 'cold'
    elif weather_type.lower() in ['rain', 'rainy', 'thunderstorm']:
        return 'rainy'
    else:
        return 'clear'

def calculate_time_of_day_demand(hour):
    if 6 <= hour < 9:
        return {'breakfast': 0.9, 'coffee': 0.8, 'light': 0.7}
    elif 9 <= hour < 12:
        return {'snacks': 0.6, 'coffee': 0.5}
    elif 12 <= hour < 14:
        return {'lunch': 0.95, 'thali': 0.9, 'rice': 0.85}
    elif 14 <= hour < 17:
        return {'snacks': 0.7, 'tea': 0.8, 'pastries': 0.6}
    elif 17 <= hour < 19:
        return {'street_food': 0.8, 'snacks': 0.85, 'tea': 0.7}
    elif 19 <= hour < 22:
        return {'dinner': 0.95, 'biryani': 0.85, 'gravy': 0.8}
    else:
        return {'late_night': 0.6, 'pizza': 0.7, 'noodles': 0.65}

def calculate_day_demand():
    today = datetime.now()
    is_weekend = today.weekday() >= 5
    
    if is_weekend:
        return {'casual': 0.9, 'family_dining': 0.85, 'desserts': 0.8, 'alcohol': 0.75}
    else:
        return {'quick_meals': 0.8, 'salads': 0.6, 'coffee': 0.85, 'lunch_boxes': 0.9}

def get_festival_boost(month, day):
    festivals = FESTIVAL_CALENDAR.get(month, [])
    if not festivals:
        return 1.0
    
    boost = 1.2
    if day in range(max(1, day-3), min(32, day+3)):
        boost = 1.35
    
    return boost

def get_real_weather(latitude, longitude):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=auto"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            return {
                'temperature': current.get('temperature_2m', 25),
                'humidity': current.get('relative_humidity_2m', 60),
                'weather_code': current.get('weather_code', 0)
            }
    except Exception as e:
        logging.warning(f"Weather API error: {e}")
    return {'temperature': 25, 'humidity': 60, 'weather_code': 0}

def decode_weather_code(code):
    weather_map = {
        0: 'clear', 1: 'partly_cloudy', 2: 'partly_cloudy', 3: 'overcast',
        45: 'foggy', 48: 'foggy', 51: 'rainy', 53: 'rainy', 55: 'rainy',
        61: 'rainy', 63: 'rainy', 65: 'rainy', 80: 'rainy', 81: 'rainy',
        82: 'rainy', 71: 'snow', 73: 'snow', 75: 'snow', 77: 'snow'
    }
    return weather_map.get(code, 'clear')

def get_real_google_trends(keywords, region_country):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, cat=71, timeframe='now 7-d', geo=region_country)
        interest_data = pytrends.interest_over_time()
        
        if not interest_data.empty:
            latest = interest_data.iloc[-1]
            trends_dict = {}
            for keyword in keywords:
                if keyword in latest.index:
                    score = latest[keyword] / 100.0
                    trends_dict[keyword.lower()] = max(0.3, score)
            return trends_dict
    except Exception as e:
        logging.warning(f"Trends API error: {e}")
    return {}

def get_real_holidays(region_country, year, month):
    try:
        country_holidays = holidays.country_holidays(region_country, years=year)
        month_holidays = []
        for date, name in sorted(country_holidays.items()):
            if date.month == month:
                month_holidays.append(name)
        return month_holidays
    except Exception as e:
        logging.warning(f"Holidays API error: {e}")
    return []

@app.route('/')
def index():
    return render_template('market.html')

@app.route('/api/forecast', methods=['POST'])
def forecast():
    data = request.json
    city = data.get('city', 'Delhi')
    region = data.get('region', 'North India')
    latitude = data.get('latitude', 28.7041)
    longitude = data.get('longitude', 77.1025)
    
    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    year = now.year
    
    season = get_season(month)
    seasonal_foods = SEASONAL_FOODS.get(season, [])
    
    local_foods = LOCAL_TASTE_PATTERNS.get(region, [])
    
    time_demand = calculate_time_of_day_demand(hour)
    day_demand = calculate_day_demand()
    
    real_weather = get_real_weather(latitude, longitude)
    temp = real_weather['temperature']
    humidity = real_weather['humidity']
    weather_code = real_weather['weather_code']
    weather_condition = decode_weather_code(weather_code)
    
    region_country = REGION_COUNTRIES.get(region, 'IN')
    real_holidays = get_real_holidays(region_country, year, month)
    
    festival_boost = 1.0
    if real_holidays:
        festival_boost = 1.25
    
    weather_foods = WEATHER_DEMAND_MAP.get(weather_condition, [])
    
    all_foods = list(set(seasonal_foods + local_foods + weather_foods))
    trends_data = get_real_google_trends(all_foods, region_country)
    
    forecast_result = {
        'city': city,
        'region': region,
        'timestamp': now.isoformat(),
        'season': season,
        'weather': weather_condition,
        'temperature': round(temp, 1),
        'humidity': humidity,
        'forecast': {
            'seasonal_demand': {
                'foods': seasonal_foods,
                'demand_level': 0.8,
                'reason': f'Popular in {season}'
            },
            'local_taste': {
                'foods': local_foods,
                'demand_level': 0.85,
                'reason': f'Local preference in {region}'
            },
            'weather_based': {
                'foods': weather_foods,
                'demand_level': 0.8,
                'reason': f'{weather_condition.capitalize()} weather preference'
            },
            'time_of_day': {
                'demand_pattern': time_demand,
                'reason': f'Popular at {hour}:00 hours'
            },
            'day_pattern': {
                'demand_pattern': day_demand,
                'reason': 'Weekend' if now.weekday() >= 5 else 'Weekday'
            },
            'festival_boost': {
                'multiplier': festival_boost,
                'holidays': real_holidays[:3],
                'reason': f'Festival/Holiday: {", ".join(real_holidays[:2])}' if real_holidays else 'Regular period'
            },
            'combined_demand': []
        }
    }
    
    for food in all_foods:
        trend_score = trends_data.get(food.lower(), 0.5)
        combined_demand = (0.25 * 0.8 + 0.25 * 0.85 + 0.25 * 0.8) * trend_score * festival_boost
        
        forecast_result['forecast']['combined_demand'].append({
            'food': food,
            'demand_score': round(min(combined_demand, 1.0), 3),
            'trend_score': round(trend_score, 3),
            'sources': ['seasonal', 'local_taste', 'weather', 'trends']
        })
    
    forecast_result['forecast']['combined_demand'].sort(
        key=lambda x: x['demand_score'],
        reverse=True
    )
    
    return jsonify(forecast_result)

@app.route('/api/detailed-forecast', methods=['POST'])
def detailed_forecast():
    data = request.json
    city = data.get('city', 'Delhi')
    region = data.get('region', 'North India')
    
    now = datetime.now()
    forecasts_24h = []
    
    for i in range(24):
        hour = (now.hour + i) % 24
        forecast_time = now + timedelta(hours=i)
        
        time_demand = calculate_time_of_day_demand(hour)
        
        season = get_season(forecast_time.month)
        seasonal_foods = SEASONAL_FOODS.get(season, [])
        local_foods = LOCAL_TASTE_PATTERNS.get(region, [])
        
        forecasts_24h.append({
            'hour': hour,
            'time': forecast_time.isoformat(),
            'demand': time_demand,
            'foods': list(set(seasonal_foods + local_foods))[:5]
        })
    
    return jsonify({
        'city': city,
        'region': region,
        'forecast_24h': forecasts_24h
    })

@app.route('/api/trends', methods=['POST'])
def get_trends():
    data = request.json
    city = data.get('city', 'Delhi')
    region = data.get('region', 'North India')
    
    local_foods = LOCAL_TASTE_PATTERNS.get(region, [])
    region_country = REGION_COUNTRIES.get(region, 'IN')
    
    trends_data = get_real_google_trends(local_foods, region_country)
    
    trends_result = {
        'city': city,
        'region': region,
        'source': 'Real Google Trends Data',
        'trending_foods': []
    }
    
    for food in local_foods[:5]:
        score = trends_data.get(food.lower(), 0.5)
        trends_result['trending_foods'].append({
            'food': food,
            'trend_score': round(score, 3),
            'direction': 'up' if score > 0.7 else 'stable' if score > 0.4 else 'down'
        })
    
    return jsonify(trends_result)

@app.route('/api/regions', methods=['GET'])
def get_regions():
    return jsonify({
        'regions': list(LOCAL_TASTE_PATTERNS.keys())
    })

@app.route('/api/location', methods=['GET'])
def get_location():
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            loc_data = response.json()
            return jsonify({
                'city': loc_data.get('city', 'Delhi'),
                'latitude': loc_data.get('latitude', 28.7041),
                'longitude': loc_data.get('longitude', 77.1025),
                'country': loc_data.get('country_name', 'India')
            })
    except Exception as e:
        logging.warning(f"Location API error: {e}")
    
    return jsonify({
        'city': 'Delhi',
        'latitude': 28.7041,
        'longitude': 77.1025,
        'country': 'India'
    })

@app.route('/api/weather', methods=['GET'])
def get_weather():
    latitude = request.args.get('latitude', 28.7041, type=float)
    longitude = request.args.get('longitude', 77.1025, type=float)
    
    real_weather = get_real_weather(latitude, longitude)
    weather_condition = decode_weather_code(real_weather['weather_code'])
    
    return jsonify({
        'temperature': round(real_weather['temperature'], 1),
        'humidity': real_weather['humidity'],
        'weather_condition': weather_condition,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
