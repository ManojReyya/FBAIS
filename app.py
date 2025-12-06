import os
import json
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from datetime import datetime, timedelta
import sqlite3
import re
from pathlib import Path
import hashlib
import secrets
from functools import wraps
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'machine_learning'))

from features.personas import CustomerPersona
from machine_learning.prediction_pipeline import ProfitabilityPredictor
from features.demand import calculate_demand_score, get_recommendations as get_demand_recommendations, INDIA_CITIES, FOOD_KEYWORDS_BY_CATEGORY
from machine_learning.ml_persona_predictor import MLPersonaPredictor
from pytrends.request import TrendReq
import requests
import holidays
import logging
from collections import defaultdict

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

if GEMINI_AVAILABLE:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBJR_G4wF60WsWguxaWv7rk7GX6Gv5_AtE')
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

app.config['DEBUG'] = True
app.config['DATA_PATH'] = 'data/final_dataset.csv'
app.config['DATABASE'] = 'data/waste_tracker.db'
app.config['USER_DATABASE'] = 'data/users.db'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

df = None
predictor = None
cities_data = None
ml_persona_predictor = None
establishments_data = None
cuisines_data = None
indian_df = None


def load_all_data():
    global df, predictor, cities_data, establishments_data, cuisines_data, ml_persona_predictor, indian_df
    try:
        df = pd.read_csv('data/final_dataset.csv')
        df['establishment_type'] = df['establishment_type'].fillna('Unknown')
        df['cuisines'] = df['cuisines'].fillna('Unknown')
        
        predictor = ProfitabilityPredictor()
        
        # Initialize ML Persona Predictor
        try:
            ml_persona_predictor = MLPersonaPredictor()
            print("[+] ML Persona Predictor initialized successfully")
        except Exception as e:
            print(f"[-] Error initializing ML Persona Predictor: {e}")
            ml_persona_predictor = None
        
        with open('data/cities_data.json', 'r') as f:
            cities_data = json.load(f)
        with open('data/establishments_data.json', 'r') as f:
            establishments_data = json.load(f)
        with open('data/cuisines_data.json', 'r') as f:
            cuisines_data = json.load(f)
        
        indian_df = pd.read_csv('data/IndianFoodDatasetCSV.csv')
        print("[+] Indian Food Dataset loaded successfully")
        
        waste_init_db()
        user_init_db()
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False


def waste_init_db():
    if not Path(app.config['DATABASE']).exists():
        conn = waste_get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE waste_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                category TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT NOT NULL,
                date_recorded DATE NOT NULL,
                cost_value REAL NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE waste_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                average_cost_per_unit REAL DEFAULT 0
            )
        ''')
        
        categories = [
            ('Vegetables', 2.5),
            ('Fruits', 3.0),
            ('Dairy', 4.5),
            ('Meat', 8.0),
            ('Bread & Grains', 2.0),
            ('Condiments', 3.5),
            ('Other', 2.0)
        ]
        
        for cat, cost in categories:
            cursor.execute('INSERT INTO waste_categories (name, average_cost_per_unit) VALUES (?, ?)', (cat, cost))
        
        conn.commit()
        conn.close()


def waste_get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def user_init_db():
    if not Path(app.config['USER_DATABASE']).exists():
        conn = user_get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                business_name TEXT,
                phone TEXT,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()


def user_get_db():
    conn = sqlite3.connect(app.config['USER_DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password, password_hash):
    return hash_password(password) == password_hash


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    if 'user_id' in session:
        conn = user_get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
        user = cursor.fetchone()
        conn.close()
        return user
    return None


@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('dashboard.html')
    return render_template('landing.html')


@app.route('/landing')
def landing():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')


@app.route('/signin')
def signin():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('signin.html')


@app.route('/signup')
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('signup.html')


@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    return render_template('dashboard.html', user=user)


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password are required'}), 400
    
    try:
        conn = user_get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and verify_password(password, user['password_hash']):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            
            conn = user_get_db()
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                          (datetime.now(), user['id']))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    business = data.get('business', '').strip()
    phone = data.get('phone', '').strip()
    password = data.get('password', '').strip()
    
    if not all([name, email, password]):
        return jsonify({'success': False, 'error': 'Name, email, and password are required'}), 400
    
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    
    if '@' not in email:
        return jsonify({'success': False, 'error': 'Invalid email format'}), 400
    
    try:
        conn = user_get_db()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (name, email, business_name, phone, password_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, business, phone, password_hash))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        session['user_id'] = user_id
        session['user_name'] = name
        session['user_email'] = email
        
        return jsonify({'success': True, 'message': 'Account created successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Email already registered'}), 409
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logout')
def api_logout():
    session.clear()
    return redirect(url_for('landing'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))


@app.route('/competitor-intelligence')
@login_required
def competitor_intelligence():
    return render_template('comp.html')


@app.route('/customer-personas')
@login_required
def customer_personas():
    return render_template('cust.html')


@app.route('/inventory-recipes')
@login_required
def inventory_recipes():
    return render_template('inventory.html')


@app.route('/location-recommendations')
@login_required
def location_recommendations():
    return render_template('locrecmd.html')


@app.route('/ai-menu-suggestions')
@login_required
def ai_menu_suggestions():
    return render_template('ai_menu.html')


@app.route('/demand-forecast')
@login_required
def demand_forecast():
    return render_template('demand.html')


@app.route('/market-forecast')
@login_required
def market_forecast():
    return render_template('market.html')


@app.route('/profitability-prediction')
@login_required
def profitability_prediction():
    return render_template('profit.html')


@app.route('/waste-tracker')
@login_required
def waste_tracker():
    return render_template('waste.html')


def parse_establishment_type(est_type_str):
    if pd.isna(est_type_str) or est_type_str == '':
        return 'Unknown'
    try:
        est_list = eval(est_type_str)
        return est_list[0] if est_list else 'Unknown'
    except:
        return 'Unknown'


def get_establishment_types():
    types_list = df['establishment_type'].apply(parse_establishment_type)
    return types_list.value_counts().head(15)


def get_city_distribution():
    return df['city'].value_counts().head(10)


def get_rating_distribution():
    return df['aggregate_rating'].value_counts().sort_index()


def get_establishment_by_city():
    city_est = df.groupby('city')['establishment'].nunique().sort_values(ascending=False).head(10)
    return city_est


def get_profitability_by_type():
    df_copy = df.copy()
    df_copy['est_type'] = df_copy['establishment_type'].apply(parse_establishment_type)
    profit_by_type = df_copy.groupby('est_type')['profitability_score'].mean().sort_values(ascending=False).head(10)
    return profit_by_type


def get_rating_vs_competitors():
    data = df.groupby('city').agg({
        'aggregate_rating': 'mean',
        'comp_count_1km': 'mean'
    }).sort_values('aggregate_rating', ascending=False).head(10)
    return data


def get_cost_analysis():
    city_cost = df.groupby('city')['average_cost_for_two'].mean().sort_values(ascending=False).head(10)
    return city_cost


def get_cuisine_distribution():
    all_cuisines = []
    for cuisines_str in df['cuisines'].dropna():
        if isinstance(cuisines_str, str):
            cuisines = [c.strip() for c in cuisines_str.split(',')]
            all_cuisines.extend(cuisines)
    cuisine_series = pd.Series(all_cuisines)
    return cuisine_series.value_counts().head(15)


def create_pie_chart(data, title):
    fig = go.Figure(data=[go.Pie(
        labels=data.index,
        values=data.values,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_bar_chart(data, title, xaxis_label='', yaxis_label='', orientation='v'):
    if orientation == 'h':
        fig = go.Figure(data=[go.Bar(
            y=data.index,
            x=data.values,
            orientation='h',
            marker=dict(color=data.values, colorscale='Viridis'),
            hovertemplate='<b>%{y}</b><br>' + yaxis_label + ': %{x:.2f}<extra></extra>'
        )])
    else:
        fig = go.Figure(data=[go.Bar(
            x=data.index,
            y=data.values,
            marker=dict(color=data.values, colorscale='Viridis'),
            hovertemplate='<b>%{x}</b><br>' + yaxis_label + ': %{y:.2f}<extra></extra>'
        )])
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_label,
        height=500,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_scatter_chart(x_data, y_data, title, xaxis_label='', yaxis_label=''):
    fig = go.Figure(data=[go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            size=10,
            color=y_data,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title=yaxis_label)
        ),
        hovertemplate='<b>%{x}</b><br>' + yaxis_label + ': %{y:.2f}<extra></extra>'
    )])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_label,
        height=500,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_heatmap_data():
    city_est = df['city'].unique()[:10]
    df_subset = df[df['city'].isin(city_est)]
    
    df_subset = df_subset.copy()
    df_subset['est_type'] = df_subset['establishment_type'].apply(parse_establishment_type)
    pivot_data = df_subset.pivot_table(
        values='aggregate_rating',
        index='est_type',
        columns='city',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='YlOrRd',
        hovertemplate='City: %{x}<br>Establishment: %{y}<br>Avg Rating: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Rating Heatmap: Establishment Type vs City',
        xaxis_title='City',
        yaxis_title='Establishment Type',
        height=500
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_competitor_heatmap():
    city_est = df['city'].unique()[:10]
    df_subset = df[df['city'].isin(city_est)]
    
    df_subset = df_subset.copy()
    df_subset['est_type'] = df_subset['establishment_type'].apply(parse_establishment_type)
    pivot_data = df_subset.pivot_table(
        values='comp_count_1km',
        index='est_type',
        columns='city',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Blues',
        hovertemplate='City: %{x}<br>Establishment: %{y}<br>Avg Competitors (1km): %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Competitor Count Heatmap: Establishment Type vs City',
        xaxis_title='City',
        yaxis_title='Establishment Type',
        height=500
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/api/charts')
def get_charts():
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        est_types = get_establishment_types()
        city_dist = get_city_distribution()
        rating_dist = get_rating_distribution()
        profit_by_type = get_profitability_by_type()
        rating_vs_comp = get_rating_vs_competitors()
        cost_analysis = get_cost_analysis()
        cuisine_dist = get_cuisine_distribution()
        
        charts = {
            'establishment_types_pie': create_pie_chart(est_types, 'Distribution of Establishment Types'),
            'cities_pie': create_pie_chart(city_dist, 'Distribution Across Cities'),
            'rating_bar': create_bar_chart(rating_dist, 'Rating Distribution', 'Rating', 'Count'),
            'profit_by_type': create_bar_chart(profit_by_type, 'Average Profitability by Establishment Type', 'Establishment Type', 'Profitability Score', 'h'),
            'cost_by_city': create_bar_chart(cost_analysis, 'Average Cost for Two by City', 'City', 'Cost', 'h'),
            'cuisine_dist': create_bar_chart(cuisine_dist, 'Top Cuisines', 'Cuisine', 'Count'),
            'rating_vs_competitors': create_scatter_chart(
                rating_vs_comp.index,
                rating_vs_comp['aggregate_rating'],
                'Average Rating vs Competitor Count by City',
                'City',
                'Average Rating'
            ),
            'heatmap_rating': create_heatmap_data(),
            'heatmap_competitors': create_competitor_heatmap()
        }
        
        stats = {
            'total_establishments': len(df),
            'total_cities': df['city'].nunique(),
            'avg_rating': float(df['aggregate_rating'].mean()),
            'avg_cost': float(df['average_cost_for_two'].mean()),
            'total_cuisines': df['cuisines'].nunique(),
            'avg_competitors_1km': float(df['comp_count_1km'].mean()),
            'avg_profitability': float(df['profitability_score'].mean()),
            'total_votes': int(df['votes'].sum())
        }
        
        return jsonify({'charts': charts, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cities', methods=['GET'])
def api_get_cities():
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        cities_list = sorted(df['city'].unique().tolist())
        return jsonify(cities_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/establishment-types', methods=['GET'])
def api_get_establishment_types():
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        types_list = df['establishment_type'].apply(parse_establishment_type).unique().tolist()
        types_list = [t for t in types_list if t != 'Unknown']
        return jsonify(sorted(types_list))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/filter-data', methods=['POST'])
def api_filter_data():
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        data = request.get_json()
        city = data.get('city', '')
        establishment_type = data.get('establishment_type', '')
        min_rating = float(data.get('min_rating', 0))
        max_cost = float(data.get('max_cost', 100000))
        
        filtered_df = df.copy()
        
        if city:
            filtered_df = filtered_df[filtered_df['city'] == city]
        
        if establishment_type:
            filtered_df = filtered_df[filtered_df['establishment_type'].apply(parse_establishment_type) == establishment_type]
        
        filtered_df = filtered_df[
            (filtered_df['aggregate_rating'] >= min_rating) &
            (filtered_df['average_cost_for_two'] <= max_cost)
        ]
        
        return jsonify({
            'count': len(filtered_df),
            'avg_rating': float(filtered_df['aggregate_rating'].mean()) if len(filtered_df) > 0 else 0,
            'avg_cost': float(filtered_df['average_cost_for_two'].mean()) if len(filtered_df) > 0 else 0,
            'avg_competitors': float(filtered_df['comp_count_1km'].mean()) if len(filtered_df) > 0 else 0,
            'avg_profitability': float(filtered_df['profitability_score'].mean()) if len(filtered_df) > 0 else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detailed-analysis', methods=['GET'])
def api_detailed_analysis():
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        city = request.args.get('city', '')
        
        if city:
            city_df = df[df['city'] == city]
            if city_df.empty:
                return jsonify({'error': 'City not found'}), 404
            title = f'Rating by Establishment Type - {city}'
        else:
            city_df = df
            title = 'Rating by Establishment Type - All Cities'
        
        city_df_copy = city_df.copy()
        city_df_copy['est_type'] = city_df_copy['establishment_type'].apply(parse_establishment_type)
        rating_by_type = city_df_copy.groupby('est_type')['aggregate_rating'].mean().sort_values(ascending=False).head(15)
        
        fig = go.Figure(data=[go.Bar(
            x=rating_by_type.index,
            y=rating_by_type.values,
            marker=dict(color=rating_by_type.values, colorscale='Viridis'),
            hovertemplate='<b>%{x}</b><br>Avg Rating: %{y:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title='Establishment Type',
            yaxis_title='Average Rating',
            height=500,
            hovermode='closest'
        )
        
        return jsonify({'rating_by_type': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/india-map', methods=['POST'])
def api_india_map():
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        filters = request.get_json() or {}
        city = filters.get('city', '')
        establishment_type = filters.get('establishment_type', '')
        
        filtered_df = df.copy()
        
        if city:
            filtered_df = filtered_df[filtered_df['city'] == city]
        
        if establishment_type:
            filtered_df = filtered_df[filtered_df['establishment_type'].apply(parse_establishment_type) == establishment_type]
        
        city_analysis = filtered_df.groupby('city').agg({
            'latitude': 'first',
            'longitude': 'first',
            'aggregate_rating': 'mean',
            'comp_count_1km': 'mean',
            'average_cost_for_two': 'mean',
            'profitability_score': 'mean',
            'establishment': 'count'
        }).reset_index()
        city_analysis.columns = ['city', 'latitude', 'longitude', 'avg_rating', 'avg_competitors', 'avg_cost', 'avg_profitability', 'count']
        
        colors_map = []
        sizes = []
        hover_texts = []
        
        for idx, row in city_analysis.iterrows():
            rating = row['avg_rating']
            if rating < 3.0:
                colors_map.append('red')
            elif rating < 3.8:
                colors_map.append('yellow')
            else:
                colors_map.append('green')
            
            sizes.append(max(10, min(40, row['count'] / 2)))
            hover_text = f"<b>{row['city']}</b><br>Rating: {rating:.2f}<br>Competitors: {row['avg_competitors']:.1f}<br>Establishments: {int(row['count'])}<br>Avg Cost: ₹{int(row['avg_cost'])}"
            hover_texts.append(hover_text)
        
        fig = go.Figure(data=[go.Scattergeo(
            lon=city_analysis['longitude'],
            lat=city_analysis['latitude'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors_map,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=city_analysis['city'],
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        )])
        
        fig.update_layout(
            title='India Performance Heatmap - Locality Analysis<br><small>🔴 Red: Poor (< 3.0) | 🟡 Yellow: Good (3.0-3.8) | 🟢 Green: Excellent (> 3.8)</small>',
            geo=dict(
                scope='asia',
                center=dict(lat=20, lon=78),
                projection_type='natural earth',
                bgcolor='rgba(230, 240, 250, 0.5)',
                coastlinecolor='lightgray',
                countrycolor='lightgray',
                countrywidth=0.5,
                lakecolor='white',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                oceancolor='rgb(204, 229, 255)'
            ),
            height=600,
            hovermode='closest'
        )
        
        return jsonify({'map': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/personas', methods=['GET'])
def get_personas():
    personas = CustomerPersona.get_all_personas()
    return jsonify(personas)


@app.route('/api/identify', methods=['POST'])
def identify_customer():
    global ml_persona_predictor
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    try:
        # Use ML predictor if available
        if ml_persona_predictor and ml_persona_predictor.model_available:
            result = ml_persona_predictor.predict_hybrid(data, ml_threshold=0.70)
        else:
            # Fallback to rule-based
            result = CustomerPersona.identify_persona(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    global ml_persona_predictor
    try:
        if ml_persona_predictor and ml_persona_predictor.model_available:
            metadata = ml_persona_predictor.metadata or {}
            return jsonify({
                'status': 'AVAILABLE',
                'type': metadata.get('model_type', 'RandomForestClassifier'),
                'accuracy': '99.95%',
                'n_estimators': metadata.get('n_estimators', 200),
                'training_samples': metadata.get('training_samples', 10500),
                'version': metadata.get('model_version', '1.0'),
                'classes': metadata.get('classes', []),
                'message': 'ML Persona Classification Model is active'
            })
        else:
            return jsonify({
                'status': 'UNAVAILABLE',
                'message': 'Using rule-based persona identification'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/<persona_key>', methods=['GET'])
def get_recommendations(persona_key):
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


@app.route('/api/recipes', methods=['GET'])
def get_recipes():
    if indian_df is None:
        return jsonify({'error': 'Recipe data not loaded'}), 500
    recipes = indian_df['RecipeName'].dropna().unique().tolist()
    recipes.sort()
    return jsonify(recipes)


@app.route('/api/recipe-details/<recipe_name>', methods=['GET'])
def get_recipe_details(recipe_name):
    try:
        if indian_df is None:
            return jsonify({'error': 'Recipe data not loaded'}), 500
        
        recipe_data = indian_df[indian_df['RecipeName'] == recipe_name]
        if recipe_data.empty:
            return jsonify({'error': 'Recipe not found'}), 404
        
        recipe = recipe_data.iloc[0]
        
        ingredients_str = recipe['Ingredients']
        servings = recipe['Servings']
        
        ingredients_list = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]
        
        return jsonify({
            'recipeName': recipe['RecipeName'],
            'originalServings': int(servings),
            'ingredients': ingredients_list,
            'prepTime': int(recipe['PrepTimeInMins']),
            'cookTime': int(recipe['CookTimeInMins']),
            'totalTime': int(recipe['TotalTimeInMins']),
            'cuisine': recipe['Cuisine'],
            'course': recipe['Course'],
            'diet': recipe['Diet']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scale-ingredients', methods=['POST'])
def scale_ingredients():
    data = request.json
    ingredients = data.get('ingredients', [])
    original_servings = data.get('originalServings', 1)
    target_servings = data.get('targetServings', 1)
    
    scaling_factor = target_servings / original_servings if original_servings > 0 else 1
    
    scaled_ingredients = []
    for ingredient in ingredients:
        scaled_ing = scale_ingredient(ingredient, scaling_factor)
        scaled_ingredients.append(scaled_ing)
    
    return jsonify({'scaledIngredients': scaled_ingredients})


def scale_ingredient(ingredient, factor):
    ingredient = ingredient.strip()
    
    number_pattern = r'(\d+\.?\d*|\d+/\d+)\s*'
    match = re.match(number_pattern, ingredient)
    
    if match:
        quantity_str = match.group(1)
        rest_of_ingredient = ingredient[len(match.group(0)):].strip()
        
        if '/' in quantity_str:
            parts = quantity_str.split('/')
            quantity = (float(parts[0]) / float(parts[1])) * factor
        else:
            quantity = float(quantity_str) * factor
        
        if quantity == int(quantity):
            return f"{int(quantity)} {rest_of_ingredient}"
        else:
            return f"{quantity:.2f} {rest_of_ingredient}".rstrip('0').rstrip('.')
    
    return ingredient


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
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/cities', methods=['GET'])
def get_cities():
    cities_list = sorted(list(cities_data.keys()))
    return jsonify({
        'cities': cities_list,
        'count': len(cities_list)
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


@app.route('/api/waste', methods=['GET'])
def get_waste():
    conn = waste_get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM waste_entries ORDER BY date_recorded DESC')
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(entries)


@app.route('/api/waste', methods=['POST'])
def add_waste():
    data = request.json
    conn = waste_get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO waste_entries 
        (item_name, category, quantity, unit, date_recorded, cost_value, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['itemName'],
        data['category'],
        float(data['quantity']),
        data['unit'],
        data['dateRecorded'],
        float(data['costValue']),
        data.get('notes', '')
    ))
    
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    
    return jsonify({'success': True, 'id': entry_id}), 201


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    conn = waste_get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM waste_entries')
    entries = [dict(row) for row in cursor.fetchall()]
    
    total_waste_cost = sum(entry['cost_value'] for entry in entries)
    total_quantity = sum(entry['quantity'] for entry in entries)
    
    category_breakdown = {}
    for entry in entries:
        category = entry['category']
        if category not in category_breakdown:
            category_breakdown[category] = {'quantity': 0, 'cost': 0, 'items': 0}
        category_breakdown[category]['quantity'] += entry['quantity']
        category_breakdown[category]['cost'] += entry['cost_value']
        category_breakdown[category]['items'] += 1
    
    last_7_days = datetime.now().date() - timedelta(days=7)
    cursor.execute('SELECT * FROM waste_entries WHERE date_recorded >= ?', (last_7_days,))
    recent_entries = [dict(row) for row in cursor.fetchall()]
    weekly_cost = sum(entry['cost_value'] for entry in recent_entries)
    
    daily_breakdown = {}
    for entry in recent_entries:
        date = entry['date_recorded']
        if date not in daily_breakdown:
            daily_breakdown[date] = {'cost': 0, 'quantity': 0}
        daily_breakdown[date]['cost'] += entry['cost_value']
        daily_breakdown[date]['quantity'] += entry['quantity']
    
    conn.close()
    
    return jsonify({
        'totalWasteCost': round(total_waste_cost, 2),
        'totalQuantity': round(total_quantity, 2),
        'entryCount': len(entries),
        'categoryBreakdown': category_breakdown,
        'weeklyCost': round(weekly_cost, 2),
        'dailyBreakdown': daily_breakdown,
        'averageItemCost': round(total_waste_cost / len(entries), 2) if entries else 0
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    conn = waste_get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM waste_categories')
    categories = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(categories)


@app.route('/api/waste/<int:waste_id>', methods=['DELETE'])
def delete_waste(waste_id):
    conn = waste_get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM waste_entries WHERE id = ?', (waste_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/export', methods=['GET'])
def export_data():
    conn = waste_get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM waste_entries ORDER BY date_recorded DESC')
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(entries)


@app.route('/api/menu-suggestions', methods=['POST'])
def generate_menu_suggestions():
    try:
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Gemini API not available'}), 500
        
        data = request.get_json()
        location = data.get('location', '')
        business_type = data.get('businessType', 'restaurant')
        
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        
        city_data = cities_data.get(location, {})
        
        local_cuisines = city_data.get('local_cuisines', ['Indian', 'North Indian', 'South Indian'])
        top_dishes = city_data.get('top_dishes', ['Biryani', 'Samosa', 'Dosa', 'Idli', 'Butter Chicken'])
        avg_budget = city_data.get('avg_budget', city_data.get('avg_cost', 300))
        
        if not top_dishes:
            top_dishes = ['Biryani', 'Samosa', 'Dosa', 'Idli', 'Butter Chicken', 'Paneer Tikka']
        if not local_cuisines:
            local_cuisines = ['Indian', 'North Indian']
        
        prompt = f"""Generate a comprehensive menu for a {business_type} in {location}, India.
        
Local preferences: {', '.join(local_cuisines)}
Popular dishes: {', '.join(top_dishes[:5])}
Average budget: ₹{avg_budget}

Provide menu items in this JSON format:
{{
    "appetizers": [
        {{"name": "item_name", "description": "brief description", "price": price_number, "isLocal": true/false}},
        ...
    ],
    "main_courses": [
        {{"name": "item_name", "description": "brief description", "price": price_number, "isLocal": true/false}},
        ...
    ],
    "sides": [
        {{"name": "item_name", "description": "brief description", "price": price_number, "isLocal": true/false}},
        ...
    ],
    "beverages": [
        {{"name": "item_name", "description": "brief description", "price": price_number, "isLocal": true/false}},
        ...
    ],
    "desserts": [
        {{"name": "item_name", "description": "brief description", "price": price_number, "isLocal": true/false}},
        ...
    ]
}}

Make sure:
1. Include 4-5 items per category
2. Prices are realistic for {location}
3. Include popular local items
4. Consider seasonal availability
5. Mix of vegetarian and non-vegetarian options
6. All prices should be between {avg_budget//2} and {avg_budget*2} rupees

Return ONLY the JSON object, no other text."""
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        menu = json.loads(response_text)
        
        all_items = []
        for category in menu.values():
            if isinstance(category, list):
                all_items.extend(category)
        
        avg_price = sum(item.get('price', 0) for item in all_items) / len(all_items) if all_items else avg_budget
        profit_margin = 60 if business_type == 'restaurant' else 50 if business_type == 'cafe' else 40
        popularity_score = min(95, 70 + len(top_dishes) * 2)
        
        insights = [
            f"Menu tailored for {location}'s local preferences",
            f"Average price point: ₹{int(avg_price)} - suitable for local market",
            "Strong focus on popular local dishes",
            "Balanced vegetarian and non-vegetarian options",
            "Items optimized for operational efficiency"
        ]
        
        seasonal_tips = [
            "Adjust seasonal vegetables based on current harvest",
            "Feature monsoon specials during rainy season",
            "Add warming beverages during winter",
            "Include cooling drinks and salads in summer",
            "Incorporate festival specials based on local calendar"
        ]
        
        return jsonify({
            'menu': menu,
            'avgPrice': int(avg_price),
            'profitMargin': profit_margin,
            'popularityScore': popularity_score,
            'insights': insights,
            'seasonal': seasonal_tips
        })
    
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid menu format generated', 'details': str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({'error': 'Error generating menu suggestions', 'details': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/profile')
@login_required
def profile():
    user = get_current_user()
    return render_template('profile.html', user=user)


@app.route('/upgrade')
@login_required
def upgrade():
    user = get_current_user()
    return render_template('upgrade.html', user=user)


@app.route('/api/update-profile', methods=['POST'])
@login_required
def api_update_profile():
    data = request.get_json()
    name = data.get('name', '').strip()
    business_name = data.get('business_name', '').strip()
    phone = data.get('phone', '').strip()
    
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    
    try:
        conn = user_get_db()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET name = ?, business_name = ?, phone = ?
            WHERE id = ?
        ''', (name, business_name, phone, session['user_id']))
        
        conn.commit()
        conn.close()
        
        session['user_name'] = name
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/forecast-demand', methods=['POST'])
def api_forecast_demand():
    try:
        data = request.get_json()
        city = data.get('city', '').strip()
        food_category = data.get('food_category', 'restaurants')
        
        if not city:
            return jsonify({'error': 'City name required'}), 400
        
        demand_data = calculate_demand_score(city, food_category)
        
        if 'error' in demand_data:
            return jsonify(demand_data), 400
        
        recommendations = get_demand_recommendations(demand_data)
        demand_data['recommendations'] = recommendations
        
        response_data = {
            'demand_score': demand_data['demand_score'],
            'category': demand_data['category'],
            'components': demand_data['components'],
            'timestamp': demand_data['timestamp'],
            'city': demand_data['city'],
            'food_category': demand_data['food_category'],
            'recommendations': recommendations
        }
        
        return jsonify(response_data), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/available-cities', methods=['GET'])
def api_available_cities():
    try:
        cities_list = list(INDIA_CITIES.keys())
        return jsonify({
            'cities': sorted(cities_list),
            'count': len(cities_list)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/food-categories', methods=['GET'])
def api_food_categories():
    try:
        categories_list = list(FOOD_KEYWORDS_BY_CATEGORY.keys())
        return jsonify({
            'categories': sorted(categories_list),
            'count': len(categories_list)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_all_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
