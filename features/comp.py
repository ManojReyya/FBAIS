import os
import json
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__, template_folder='templates')

DATA_PATH = 'final_dataset.csv'
df = None



def load_data():
    global df
    try:
        df = pd.read_csv(DATA_PATH)
        df['establishment_type'] = df['establishment_type'].fillna('Unknown')
        df['cuisines'] = df['cuisines'].fillna('Unknown')
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

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
    df['est_type'] = df['establishment_type'].apply(parse_establishment_type)
    profit_by_type = df.groupby('est_type')['profitability_score'].mean().sort_values(ascending=False).head(10)
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

def create_box_plot(data_dict, title, yaxis_label=''):
    fig = go.Figure()
    for label, values in data_dict.items():
        fig.add_trace(go.Box(y=values, name=label))
    
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_label,
        height=500,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_heatmap_data():
    city_est = df['city'].unique()[:10]
    df_subset = df[df['city'].isin(city_est)]
    
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

def get_performance_color(rating):
    if rating < 3.0:
        return 'red'
    elif rating < 3.8:
        return 'gold'
    else:
        return 'green'

def create_india_map_data(filtered_df=None):
    try:
        if filtered_df is None:
            filtered_df = df.copy()
        
        if len(filtered_df) == 0:
            filtered_df = df.copy()
        
        filtered_df = filtered_df.dropna(subset=['latitude', 'longitude', 'aggregate_rating'])
        
        if len(filtered_df) == 0:
            filtered_df = df.dropna(subset=['latitude', 'longitude', 'aggregate_rating']).copy()
        
        data_by_locality = filtered_df.groupby('locality').agg({
            'latitude': 'first',
            'longitude': 'first',
            'aggregate_rating': 'mean',
            'comp_count_1km': 'mean',
            'profitability_score': 'mean',
            'city': 'first',
            'establishment': 'count'
        }).reset_index()
        
        data_by_locality.columns = ['locality', 'latitude', 'longitude', 'avg_rating', 'avg_comp', 'avg_profit', 'city', 'est_count']
        data_by_locality = data_by_locality.fillna(0)
        
        print(f"Total localities with data: {len(data_by_locality)}")
        
        lats, lons, names, texts, colors, sizes = [], [], [], [], [], []
        
        for _, row in data_by_locality.iterrows():
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            locality = str(row['locality'])
            city = str(row['city'])
            avg_rating = float(row['avg_rating'])
            avg_comp = float(row['avg_comp'])
            est_count = int(row['est_count'])
            avg_profit = float(row['avg_profit'])
            
            if 8 <= lat <= 35 and 68 <= lon <= 97:
                lats.append(lat)
                lons.append(lon)
                names.append(locality)
                color = get_performance_color(avg_rating)
                colors.append(color)
                
                text = f"<b>{locality}, {city}</b><br>Establishments: {est_count}<br>Avg Rating: {avg_rating:.2f}<br>Performance: {'🟢 Excellent' if avg_rating >= 3.8 else '🟡 Good' if avg_rating >= 3.0 else '🔴 Needs Improvement'}<br>Competitors (1km): {avg_comp:.1f}<br>Profitability: {avg_profit:.2f}"
                texts.append(text)
                
                size = max(min(est_count / 30, 35), 6)
                sizes.append(size)
        
        print(f"Total mapped locations: {len(lats)}")
        print(f"Red (< 3.0): {sum(1 for c in colors if c == 'red')}")
        print(f"Yellow (3.0-3.8): {sum(1 for c in colors if c == 'gold')}")
        print(f"Green (> 3.8): {sum(1 for c in colors if c == 'green')}")
        
        if len(lats) == 0:
            raise Exception("No valid lat/lon coordinates found in filtered data")
        
        trace_red = go.Scattergeo(
            lat=[lats[i] for i in range(len(lats)) if colors[i] == 'red'],
            lon=[lons[i] for i in range(len(lons)) if colors[i] == 'red'],
            mode='markers',
            marker=dict(
                size=[sizes[i] for i in range(len(sizes)) if colors[i] == 'red'],
                color='red',
                opacity=0.7,
                line=dict(width=2, color='darkred')
            ),
            text=[texts[i] for i in range(len(texts)) if colors[i] == 'red'],
            name='🔴 Poor Performance (< 3.0)',
            hovertemplate='%{text}<extra></extra>'
        )
        
        trace_yellow = go.Scattergeo(
            lat=[lats[i] for i in range(len(lats)) if colors[i] == 'gold'],
            lon=[lons[i] for i in range(len(lons)) if colors[i] == 'gold'],
            mode='markers',
            marker=dict(
                size=[sizes[i] for i in range(len(sizes)) if colors[i] == 'gold'],
                color='gold',
                opacity=0.7,
                line=dict(width=2, color='orange')
            ),
            text=[texts[i] for i in range(len(texts)) if colors[i] == 'gold'],
            name='🟡 Good Performance (3.0-3.8)',
            hovertemplate='%{text}<extra></extra>'
        )
        
        trace_green = go.Scattergeo(
            lat=[lats[i] for i in range(len(lats)) if colors[i] == 'green'],
            lon=[lons[i] for i in range(len(lons)) if colors[i] == 'green'],
            mode='markers',
            marker=dict(
                size=[sizes[i] for i in range(len(sizes)) if colors[i] == 'green'],
                color='green',
                opacity=0.7,
                line=dict(width=2, color='darkgreen')
            ),
            text=[texts[i] for i in range(len(texts)) if colors[i] == 'green'],
            name='🟢 Excellent Performance (> 3.8)',
            hovertemplate='%{text}<extra></extra>'
        )
        
        fig = go.Figure(data=[trace_red, trace_yellow, trace_green])
        
        fig.update_layout(
            title='Competitor Intelligence Map - India (By Locality Performance)',
            geo=dict(
                scope='asia',
                projection_type='mercator',
                center=dict(lat=20, lon=77),
                lataxis_range=[8, 35],
                lonaxis_range=[68, 97],
                showland=True,
                landcolor='rgb(243, 243, 243)',
                showocean=True,
                oceancolor='rgb(204, 229, 255)',
                coastlinecolor='rgb(200, 220, 255)',
                showlakes=True,
                lakecolor='rgb(204, 229, 255)',
                countrycolor='rgb(200, 220, 255)',
                showcountries=True
            ),
            height=600,
            margin=dict(r=0, t=40, l=0, b=0),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error in create_india_map_data: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('comp.html')

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
            'avg_profitability': float(df['profitability_score'].mean()),
            'total_votes': int(df['votes'].sum())
        }
        
        return jsonify({
            'charts': charts,
            'stats': stats
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter-data', methods=['POST'])
def filter_data():
    try:
        filters = request.json
        filtered_df = df.copy()
        
        if 'city' in filters and filters['city']:
            filtered_df = filtered_df[filtered_df['city'] == filters['city']]
        
        if 'establishment_type' in filters and filters['establishment_type']:
            filtered_df['est_type'] = filtered_df['establishment_type'].apply(parse_establishment_type)
            filtered_df = filtered_df[filtered_df['est_type'] == filters['establishment_type']]
        
        if 'min_rating' in filters:
            filtered_df = filtered_df[filtered_df['aggregate_rating'] >= filters['min_rating']]
        
        if 'max_cost' in filters:
            filtered_df = filtered_df[filtered_df['average_cost_for_two'] <= filters['max_cost']]
        
        stats = {
            'count': len(filtered_df),
            'avg_rating': float(filtered_df['aggregate_rating'].mean()),
            'avg_cost': float(filtered_df['average_cost_for_two'].mean()),
            'avg_competitors': float(filtered_df['comp_count_1km'].mean()),
            'avg_profitability': float(filtered_df['profitability_score'].mean())
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cities')
def get_cities():
    cities = sorted(df['city'].unique().tolist())
    return jsonify(cities)

@app.route('/api/establishment-types')
def get_establishment_types_api():
    df['est_type'] = df['establishment_type'].apply(parse_establishment_type)
    types = sorted(df['est_type'].unique().tolist())
    return jsonify(types)

@app.route('/api/detailed-analysis')
def detailed_analysis():
    try:
        city = request.args.get('city')
        
        if city:
            city_df = df[df['city'] == city]
        else:
            city_df = df
        
        city_df['est_type'] = city_df['establishment_type'].apply(parse_establishment_type)
        
        analysis = {
            'top_establishments': city_df.nlargest(5, 'profitability_score')[['establishment', 'aggregate_rating', 'average_cost_for_two', 'profitability_score']].to_dict('records'),
            'establishment_type_summary': city_df.groupby('est_type').agg({
                'aggregate_rating': 'mean',
                'average_cost_for_two': 'mean',
                'comp_count_1km': 'mean'
            }).round(2).to_dict(),
            'rating_by_type': create_bar_chart(
                city_df.groupby('est_type')['aggregate_rating'].mean().sort_values(ascending=False),
                f'Average Rating by Type{"" if not city else f" - {city}"}',
                'Type',
                'Rating',
                'h'
            )
        }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/india-map', methods=['POST', 'GET'])
def india_map():
    try:
        if df is None:
            print("ERROR: Data not loaded in india_map")
            return jsonify({'error': 'Data not loaded'}), 500
            
        filters = request.get_json() if request.method == 'POST' else {}
        if filters is None:
            filters = {}
            
        print(f"India Map Request - Filters: {filters}")
        
        filtered_df = df.copy()
        print(f"Starting with {len(filtered_df)} records")
        
        if 'city' in filters and filters['city']:
            filtered_df = filtered_df[filtered_df['city'] == filters['city']]
            print(f"After city filter: {len(filtered_df)} records")
        
        if 'establishment_type' in filters and filters['establishment_type']:
            filtered_df['est_type'] = filtered_df['establishment_type'].apply(parse_establishment_type)
            filtered_df = filtered_df[filtered_df['est_type'] == filters['establishment_type']]
            print(f"After est_type filter: {len(filtered_df)} records")
        
        if 'min_rating' in filters and filters['min_rating'] and filters['min_rating'] > 0:
            filtered_df = filtered_df[filtered_df['aggregate_rating'] >= filters['min_rating']]
            print(f"After rating filter: {len(filtered_df)} records")
        
        if 'max_cost' in filters and filters['max_cost'] and filters['max_cost'] > 0:
            filtered_df = filtered_df[filtered_df['average_cost_for_two'] <= filters['max_cost']]
            print(f"After cost filter: {len(filtered_df)} records")
        
        print("Creating map data...")
        map_json = create_india_map_data(filtered_df)
        print("Map data created successfully")
        
        return jsonify({'map': map_json})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in india_map endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    if load_data():
        print(f"Data loaded successfully. Total records: {len(df)}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load data")
