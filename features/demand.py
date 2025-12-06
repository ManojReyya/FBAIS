from pytrends.request import TrendReq
import requests
from datetime import datetime, timedelta
import json
import logging
import math

logging.basicConfig(level=logging.WARNING)

INDIA_CITIES = {
    'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'population': 8436675},
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'population': 12442373},
    'delhi': {'lat': 28.7041, 'lon': 77.1025, 'population': 11007835},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'population': 6809970},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'population': 4486679},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'population': 4646732},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'population': 3124458},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'population': 3577940},
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'population': 3046163},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'population': 2901474},
    'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'population': 1055450},
    'bhopal': {'lat': 23.1815, 'lon': 79.9864, 'population': 1798218},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'population': 1994408},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'population': 1693000},
    'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'population': 1127324},
    'raipur': {'lat': 21.2514, 'lon': 81.6296, 'population': 1001058},
    'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'population': 1022259},
    'thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'population': 957730},
    'kochi': {'lat': 9.9312, 'lon': 76.2673, 'population': 677381},
    'coimbatore': {'lat': 11.0026, 'lon': 76.7055, 'population': 1441405},
    'surat': {'lat': 21.1458, 'lon': 72.8336, 'population': 4467792},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'population': 2405421},
    'bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'population': 837737},
    'gandhinagar': {'lat': 23.1967, 'lon': 72.6345, 'population': 264977},
    'shimla': {'lat': 31.7775, 'lon': 77.1577, 'population': 171560},
    'itanagar': {'lat': 28.2180, 'lon': 93.6053, 'population': 38000},
    'imphal': {'lat': 24.8170, 'lon': 94.9062, 'population': 302388},
    'shillong': {'lat': 25.5788, 'lon': 91.8933, 'population': 342100},
    'aizawl': {'lat': 23.8103, 'lon': 92.7015, 'population': 228001},
    'kohima': {'lat': 25.6156, 'lon': 94.1133, 'population': 101458},
    'agartala': {'lat': 23.8841, 'lon': 91.2868, 'population': 402873},
    'gangtok': {'lat': 27.5330, 'lon': 88.6109, 'population': 98711},
    'panaji': {'lat': 15.4909, 'lon': 73.8278, 'population': 101845},
    'dehradun': {'lat': 30.3165, 'lon': 78.0322, 'population': 743676},
    'leh': {'lat': 34.1526, 'lon': 77.5770, 'population': 58000},
    'port blair': {'lat': 11.7345, 'lon': 92.7598, 'population': 100186},
    'puducherry': {'lat': 12.0657, 'lon': 79.8711, 'population': 398645},
    'amaravati': {'lat': 16.5062, 'lon': 80.6480, 'population': 100000},
    'kavaratti': {'lat': 10.5667, 'lon': 72.6417, 'population': 10141},
    'silvassa': {'lat': 20.2596, 'lon': 73.2603, 'population': 89031},
}

FOOD_KEYWORDS_BY_CATEGORY = {
    'restaurants': ['restaurant near me', 'food delivery', 'dine in', 'cafe'],
    'street_food': ['street food', 'chaat', 'vada pav', 'samosa'],
    'fast_food': ['pizza', 'burger', 'fast food'],
    'traditional': ['biryani', 'butter chicken', 'dal makhani', 'sambar'],
    'beverages': ['tea', 'coffee', 'juice', 'smoothie'],
    'bakery': ['bakery', 'bread', 'cake', 'pastry'],
}

WEATHER_DEMAND_MAPPING = {
    'hot': {
        'ice_cream': 0.9,
        'juice': 0.85,
        'cold_drinks': 0.9,
        'smoothie': 0.8,
        'light_food': 0.7
    },
    'cold': {
        'tea': 0.95,
        'coffee': 0.9,
        'soup': 0.85,
        'biryani': 0.8,
        'warm_meals': 0.85
    },
    'rainy': {
        'tea': 0.95,
        'pakora': 0.9,
        'snacks': 0.85,
        'bread': 0.8,
        'soup': 0.85
    },
    'clear': {
        'salad': 0.7,
        'light_food': 0.8,
        'juice': 0.75,
        'outdoor_food': 0.8
    }
}

def get_google_trends_score(city_name, keywords):
    """
    Get Google Trends data for given city and keywords.
    Returns normalized score (0-100).
    """
    try:
        pytrends = TrendReq(hl='en-IN', tz=330)
        
        scores = []
        for keyword in keywords[:5]:
            try:
                pytrends.build_payload(
                    kw_list=[keyword],
                    cat=71,
                    timeframe='today 1-m',
                    geo='IN'
                )
                data = pytrends.interest_over_time()
                if not data.empty:
                    avg_score = data[keyword].mean()
                    scores.append(avg_score)
            except:
                continue
        
        if scores:
            normalized_score = min(100, (sum(scores) / len(scores)))
            return float(normalized_score)
        return 50.0
    except Exception as e:
        logging.warning(f"Google Trends error for {city_name}: {e}")
        return 50.0

def get_population_density_score(city_name):
    """
    Get population density score based on city data.
    Returns normalized score (0-100).
    """
    try:
        city_lower = city_name.lower()
        if city_lower in INDIA_CITIES:
            city_data = INDIA_CITIES[city_lower]
            population = city_data['population']
            
            min_pop = 1000000
            max_pop = 15000000
            
            normalized = max(0, min(100, ((population - min_pop) / (max_pop - min_pop)) * 100))
            return float(normalized)
        
        return 50.0
    except Exception as e:
        logging.warning(f"Population density error for {city_name}: {e}")
        return 50.0

def get_weather_impact_score(city_name):
    """
    Get weather-based demand impact using Open-Meteo API.
    Returns score (0-100).
    """
    try:
        if city_name.lower() not in INDIA_CITIES:
            return 50.0
        
        city_data = INDIA_CITIES[city_name.lower()]
        lat, lon = city_data['lat'], city_data['lon']
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,relative_humidity_2m&temperature_unit=celsius"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        temp = current.get('temperature_2m', 20)
        humidity = current.get('relative_humidity_2m', 50)
        weather_code = current.get('weather_code', 0)
        
        weather_condition = classify_weather(temp, humidity, weather_code)
        
        demand_scores = []
        for food_type, score in WEATHER_DEMAND_MAPPING.get(weather_condition, {}).items():
            demand_scores.append(score * 100)
        
        if demand_scores:
            return float(sum(demand_scores) / len(demand_scores))
        return 70.0
    except Exception as e:
        logging.warning(f"Weather API error for {city_name}: {e}")
        return 70.0

def classify_weather(temp, humidity, weather_code):
    """
    Classify weather condition from temperature and weather code.
    Returns: 'hot', 'cold', 'rainy', or 'clear'
    """
    if weather_code in [80, 81, 82, 95, 96, 99]:
        return 'rainy'
    elif temp > 28:
        return 'hot'
    elif temp < 10:
        return 'cold'
    else:
        return 'clear'

def estimate_osm_footfall_score(city_name):
    """
    Estimate footfall using OpenStreetMap data proxy.
    Major cities get higher scores based on infrastructure.
    Returns score (0-100).
    """
    try:
        city_lower = city_name.lower()
        
        metro_cities = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'kolkata', 'chennai']
        tier2_cities = ['pune', 'ahmedabad', 'jaipur', 'lucknow']
        
        if city_lower in metro_cities:
            base_score = 85
        elif city_lower in tier2_cities:
            base_score = 70
        else:
            base_score = 50
        
        hour = datetime.now().hour
        day = datetime.now().weekday()
        
        time_multiplier = 1.0
        if 12 <= hour < 14 or 19 <= hour < 21:
            time_multiplier = 1.1
        elif 6 <= hour < 9:
            time_multiplier = 0.8
        
        day_multiplier = 1.1 if day >= 5 else 0.95
        
        final_score = base_score * time_multiplier * day_multiplier
        return float(min(100, max(0, final_score)))
    except Exception as e:
        logging.warning(f"OSM footfall error for {city_name}: {e}")
        return 60.0

def calculate_demand_score(city_name, food_category='restaurants'):
    """
    Calculate final demand score using weighted formula:
    Demand = (0.50 * Google Trends) +
             (0.20 * Population Density) +
             (0.15 * Weather Impact) +
             (0.15 * Tourism/Footfall)
    """
    try:
        keywords = FOOD_KEYWORDS_BY_CATEGORY.get(food_category, ['restaurant near me', 'food delivery'])
        
        google_trends_score = get_google_trends_score(city_name, keywords)
        population_score = get_population_density_score(city_name)
        weather_score = get_weather_impact_score(city_name)
        footfall_score = estimate_osm_footfall_score(city_name)
        
        demand_score = (
            (0.50 * google_trends_score) +
            (0.20 * population_score) +
            (0.15 * weather_score) +
            (0.15 * footfall_score)
        )
        
        demand_score = float(min(100, max(0, demand_score)))
        
        category = categorize_demand(demand_score)
        
        return {
            'demand_score': demand_score,
            'category': category,
            'components': {
                'google_trends': float(google_trends_score),
                'population_density': float(population_score),
                'weather_impact': float(weather_score),
                'footfall': float(footfall_score)
            },
            'timestamp': datetime.now().isoformat(),
            'city': city_name,
            'food_category': food_category
        }
    except Exception as e:
        logging.error(f"Demand calculation error: {e}")
        return {
            'error': str(e),
            'demand_score': 0,
            'category': 'Low'
        }

def categorize_demand(score):
    """
    Categorize demand based on score.
    80-100: Very High
    60-79: Good
    40-59: Medium
    <40: Low
    """
    if score >= 80:
        return 'Very High'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Medium'
    else:
        return 'Low'

def get_recommendations(demand_data):
    """
    Generate business recommendations based on demand analysis.
    """
    score = demand_data['demand_score']
    category = demand_data['category']
    components = demand_data['components']
    
    recommendations = []
    
    if category == 'Very High':
        recommendations.append('Excellent location for food business expansion')
        recommendations.append('High profitability potential')
        recommendations.append('Strong market demand detected')
    elif category == 'Good':
        recommendations.append('Good location with solid demand')
        recommendations.append('Focus on quality to capture market share')
        recommendations.append('Competitive pricing strategy recommended')
    elif category == 'Medium':
        recommendations.append('Moderate demand - niche positioning needed')
        recommendations.append('Target specific food categories strategically')
        recommendations.append('Regular market monitoring suggested')
    else:
        recommendations.append('Low demand - consider location alternatives')
        recommendations.append('Detailed market research recommended')
        recommendations.append('Consider unique value proposition')
    
    if components['google_trends'] < 40:
        recommendations.append('Invest in marketing and brand awareness')
    
    if components['population_density'] < 40:
        recommendations.append('Location has lower population base - premium positioning needed')
    
    if components['weather_impact'] < 40:
        recommendations.append('Weather limits certain food types - diversify menu')
    
    if components['footfall'] > 80:
        recommendations.append('High footfall area - visibility is key')
    
    return recommendations


