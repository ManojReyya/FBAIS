from flask import Flask, render_template, request, jsonify
import json
import os
from pathlib import Path

app = Flask(__name__)

def load_indian_data():
    data_path = Path(__file__).parent / 'indian_data.json'
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception(f"Indian data file not found at {data_path}")
    except json.JSONDecodeError:
        raise Exception("Invalid JSON in indian_data.json")

SCORING_WEIGHTS = {
    'market_size': 0.25,
    'income_potential': 0.20,
    'tourism_demand': 0.20,
    'competition_factor': 0.20,
    'growth_potential': 0.15
}

def calculate_market_size_score(state_data):
    population = state_data.get('population', 10000000)
    population_density = state_data.get('population_density', 300)
    
    size_score = min(100, (population / 100000000) * 50)
    density_score = min(100, (population_density / 1000) * 50)
    
    return (size_score + density_score) / 2

def calculate_income_potential(state_data):
    income_level = state_data.get('city_income_level', 35000)
    population = state_data.get('population', 10000000)
    
    income_score = min(100, (income_level / 60000) * 100)
    market_capacity = min(100, (population / 50000000) * 50)
    
    return (income_score * 0.7) + (market_capacity * 0.3)

def calculate_tourism_demand(state_data):
    tourism_index = state_data.get('tourism_index', 5)
    population_density = state_data.get('population_density', 300)
    
    tourism_score = (tourism_index / 10) * 100
    foot_traffic_potential = min(100, (population_density / 500) * 50)
    
    return (tourism_score * 0.6) + (foot_traffic_potential * 0.4)

def calculate_competition_score(state_data):
    population_density = state_data.get('population_density', 300)
    population = state_data.get('population', 10000000)
    
    if population_density < 150:
        market_saturation = 20
    elif population_density < 400:
        market_saturation = 50
    elif population_density < 800:
        market_saturation = 75
    else:
        market_saturation = 95
    
    inverse_competition = 100 - market_saturation
    return max(20, inverse_competition)

def calculate_growth_potential(state_data):
    income_level = state_data.get('city_income_level', 35000)
    tourism_index = state_data.get('tourism_index', 5)
    population = state_data.get('population', 10000000)
    
    income_growth = min(100, (income_level / 50000) * 60)
    tourism_growth = (tourism_index / 10) * 40
    pop_growth = min(100, (population / 100000000) * 30)
    
    return (income_growth * 0.5) + (tourism_growth * 0.3) + (pop_growth * 0.2)

def calculate_opportunity_score(state_data, dish):
    try:
        market_size = calculate_market_size_score(state_data)
        income_potential = calculate_income_potential(state_data)
        tourism_demand = calculate_tourism_demand(state_data)
        competition_score = calculate_competition_score(state_data)
        growth_potential = calculate_growth_potential(state_data)
        
        total_score = (
            (market_size * SCORING_WEIGHTS['market_size']) +
            (income_potential * SCORING_WEIGHTS['income_potential']) +
            (tourism_demand * SCORING_WEIGHTS['tourism_demand']) +
            (competition_score * SCORING_WEIGHTS['competition_factor']) +
            (growth_potential * SCORING_WEIGHTS['growth_potential'])
        )
        
        return max(15, min(100, round(total_score, 1)))
    except Exception as e:
        return 50

def get_competition_level(state_data):
    try:
        population_density = state_data.get('population_density', 300)
        income = state_data.get('city_income_level', 35000)
        
        if population_density > 1000 and income > 45000:
            return 'Very High'
        elif population_density > 800:
            return 'High'
        elif population_density > 300:
            return 'Medium'
        else:
            return 'Low'
    except:
        return 'Medium'

def get_market_viability(state_data):
    try:
        population = state_data.get('population', 10000000)
        income = state_data.get('city_income_level', 35000)
        tourism = state_data.get('tourism_index', 5)
        
        viability_score = (population / 100000000) * 30 + (income / 50000) * 40 + (tourism / 10) * 30
        
        if viability_score >= 70:
            return 'Excellent'
        elif viability_score >= 55:
            return 'Very Good'
        elif viability_score >= 40:
            return 'Good'
        else:
            return 'Moderate'
    except:
        return 'Good'

@app.route('/')
def index():
    return render_template('locrecmd.html')

@app.route('/api/indian-data', methods=['GET'])
def get_indian_data():
    try:
        data = load_indian_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    try:
        dish = request.args.get('dish', '').strip()
        if not dish:
            return jsonify({'error': 'Dish parameter is required'}), 400

        data = load_indian_data()
        recommendations = []

        for state_name, state_data in data['states'].items():
            if dish in state_data['top_dishes']:
                capital = state_data.get('capital', 'N/A')
                population = state_data.get('population', 0)
                pop_density = state_data.get('population_density', 0)
                income_level = state_data.get('city_income_level', 0)
                tourism_index = state_data.get('tourism_index', 0)
                
                market_size_score = calculate_market_size_score(state_data)
                income_potential = calculate_income_potential(state_data)
                tourism_demand = calculate_tourism_demand(state_data)
                competition_score = calculate_competition_score(state_data)
                growth_potential = calculate_growth_potential(state_data)
                
                opportunity_score = calculate_opportunity_score(state_data, dish)
                competition_level = get_competition_level(state_data)
                viability = get_market_viability(state_data)
                
                market_reason = f"{capital} has {viability.lower()} market viability with strong demand for {dish}. Population: {population/1e6:.1f}M, Income: ₹{income_level}+, Tourism: {tourism_index}/10"
                
                recommendations.append({
                    'state': state_name,
                    'capital': capital,
                    'dish': dish,
                    'score': round(opportunity_score, 1),
                    'population': population,
                    'population_density': pop_density,
                    'income_level': income_level,
                    'tourism_index': tourism_index,
                    'competition': competition_level,
                    'viability': viability,
                    'reason': market_reason,
                    'scoring_breakdown': {
                        'market_size': round(market_size_score, 1),
                        'income_potential': round(income_potential, 1),
                        'tourism_demand': round(tourism_demand, 1),
                        'competition_factor': round(competition_score, 1),
                        'growth_potential': round(growth_potential, 1)
                    }
                })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        top_recommendation = recommendations[0] if recommendations else None
        market_summary = {
            'average_score': round(sum(r['score'] for r in recommendations) / len(recommendations), 1) if recommendations else 0,
            'highest_score': top_recommendation['score'] if top_recommendation else 0,
            'top_location': top_recommendation['capital'] if top_recommendation else None,
            'top_state': top_recommendation['state'] if top_recommendation else None
        } if recommendations else {}
        
        return jsonify({
            'locations': recommendations,
            'total_matches': len(recommendations),
            'dish_searched': dish,
            'market_summary': market_summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/all-dishes', methods=['GET'])
def get_all_dishes():
    try:
        data = load_indian_data()
        all_dishes = set()
        
        for state_data in data['states'].values():
            all_dishes.update(state_data.get('top_dishes', []))
        
        return jsonify({'dishes': sorted(list(all_dishes))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/states-list', methods=['GET'])
def get_states_list():
    try:
        data = load_indian_data()
        states = list(data['states'].keys())
        return jsonify({'states': sorted(states)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)