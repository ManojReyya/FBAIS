from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from personas import CustomerPersona
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('cust.html')

@app.route('/api/personas', methods=['GET'])
def get_personas():
    """Get all available personas"""
    personas = CustomerPersona.get_all_personas()
    return jsonify(personas)

@app.route('/api/identify', methods=['POST'])
def identify_customer():
    """Identify customer persona based on provided criteria"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        result = CustomerPersona.identify_persona(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/<persona_key>', methods=['GET'])
def get_recommendations(persona_key):
    """Get detailed recommendations for a specific persona"""
    if persona_key not in CustomerPersona.PERSONAS:
        return jsonify({'error': 'Persona not found'}), 404
    
    persona_data = CustomerPersona.PERSONAS[persona_key]
    
    recommendations = {
        'persona_key': persona_key,
        'persona_name': persona_data['name'],
        'characteristics': persona_data['characteristics'],
        'peak_times': persona_data['peak_times'],
        'what_to_sell': persona_data['products'],
        'price_strategy': persona_data['price_strategy'],
        'marketing_strategies': persona_data['marketing'],
        'average_spending': persona_data['avg_spending']
    }
    
    return jsonify(recommendations)

@app.route('/api/quick-analyze', methods=['POST'])
def quick_analyze():
    """Quick analysis based on simple inputs for food business"""
    data = request.get_json()
    
    criteria = {
        'time': int(data.get('time', datetime.now().hour)),
        'budget_level': data.get('budget_level', 'medium'),
        'food_type': data.get('food_type', 'fast_casual'),
        'occasion': data.get('occasion', 'daily_meal'),
        'customer_type': data.get('customer_type', 'working_professional'),
        'delivery_preference': data.get('delivery_preference', 'app_delivery'),
        'payment_method': data.get('payment_method', 'cash')
    }
    
    result = CustomerPersona.identify_persona(criteria)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
