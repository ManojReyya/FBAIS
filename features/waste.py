from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import sqlite3
import json
from pathlib import Path

app = Flask(__name__)
DATABASE = 'waste_tracker.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not Path(DATABASE).exists():
        conn = get_db()
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

@app.route('/')
def index():
    return render_template('waste.html')

@app.route('/api/waste', methods=['GET'])
def get_waste():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM waste_entries ORDER BY date_recorded DESC')
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(entries)

@app.route('/api/waste', methods=['POST'])
def add_waste():
    data = request.json
    conn = get_db()
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
    conn = get_db()
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
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM waste_categories')
    categories = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(categories)

@app.route('/api/waste/<int:waste_id>', methods=['DELETE'])
def delete_waste(waste_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM waste_entries WHERE id = ?', (waste_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/export', methods=['GET'])
def export_data():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM waste_entries ORDER BY date_recorded DESC')
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(entries)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
