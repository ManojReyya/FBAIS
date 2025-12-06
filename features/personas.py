import json
from typing import Dict, List, Any
from datetime import datetime

class CustomerPersona:
    PERSONAS = {
        'street_food': {
            'name': 'Street Food & Quick Bites Customers',
            'characteristics': ['Budget-conscious', 'Time-pressed', 'Casual', 'Local preference'],
            'peak_times': ['08:00-10:00', '13:00-14:30', '18:00-20:00'],
            'cuisine_types': ['Chaat', 'Samosa', 'Dosa', 'Momos', 'Vada Pav', 'Bhel Puri', 'Panipuri'],
            'products': ['Quick meals', 'Tea & coffee', 'Snacks', 'Desserts', 'Street food'],
            'avg_spending': '₹50-₹150',
            'payment_methods': ['Cash', 'UPI', 'Paytm'],
            'price_strategy': 'High volume, low margin, affordable pricing',
            'marketing': [
                'Local WhatsApp groups',
                'Instagram reels (trending food)',
                'Direct word-of-mouth',
                'Google Maps visibility',
                'Instagram/Facebook local ads',
                'Daily specials & combos'
            ]
        },
        'fast_casual': {
            'name': 'Fast Casual & QSR Customers',
            'characteristics': ['Moderate budget', 'Convenience-focused', 'Family-friendly', 'Quality-conscious'],
            'peak_times': ['12:00-14:00', '19:00-21:00', 'Weekends all day'],
            'cuisine_types': ['North Indian', 'Chinese', 'Continental', 'Hyderabadi', 'Biryani', 'Thali'],
            'products': ['Full meals', 'Combo deals', 'Beverages', 'Desserts', 'Family packs'],
            'avg_spending': '₹200-₹500',
            'payment_methods': ['Cash', 'Debit/Credit card', 'UPI', 'Mobile wallets'],
            'price_strategy': 'Value for money, portion-focused, combo discounts',
            'marketing': [
                'Instagram & YouTube',
                'Google & Zomato listings',
                'Facebook ads (Local targeting)',
                'Loyalty apps (PayBack, Swiggy)',
                'Email newsletters',
                'Student & office discounts'
            ]
        },
        'fine_dining': {
            'name': 'Fine Dining & Premium Customers',
            'characteristics': ['Luxury-oriented', 'Experience-focused', 'Occasion-based', 'Status-conscious'],
            'peak_times': ['19:00-23:00', 'Weekends evenings', 'Special occasions'],
            'cuisine_types': ['Multi-cuisine', 'Italian', 'Continental', 'Japanese', 'Premium North Indian', 'Coastal cuisine'],
            'products': ['Premium meals', 'Ambiance experience', 'Special occasion catering', 'Wine pairing', 'Chef specials'],
            'avg_spending': '₹800-₹3000+',
            'payment_methods': ['Credit card', 'UPI', 'Corporate accounts'],
            'price_strategy': 'Premium positioning, experience-based pricing, seasonal menus',
            'marketing': [
                'Instagram (aesthetic focus)',
                'Facebook events',
                'WhatsApp elite groups',
                'Food bloggers & influencers',
                'Email campaigns (High-spend customers)',
                'Corporate partnerships'
            ]
        },
        'cloud_kitchen': {
            'name': 'Cloud Kitchen & Delivery Order Customers',
            'characteristics': ['Tech-savvy', 'Busy professionals', 'Delivery-dependent', 'App users'],
            'peak_times': ['12:00-13:30', '19:00-22:00', 'Late night'],
            'cuisine_types': ['Variety of cuisines', 'Fusion food', 'Convenience meals', 'Healthy options'],
            'products': ['Quick meals', 'Ready-to-eat', 'Healthy bowls', 'Desserts', 'Beverages'],
            'avg_spending': '₹300-₹800',
            'payment_methods': ['UPI', 'Card', 'Digital wallets', 'App credits'],
            'price_strategy': 'Competitive app pricing, delivery bundles, app-exclusive discounts',
            'marketing': [
                'Zomato/Swiggy optimization',
                'Google Business listing',
                'App notifications & deals',
                'Facebook & Instagram ads',
                'Influencer partnerships (food reviewers)',
                'First-order discounts'
            ]
        },
        'regional_specialty': {
            'name': 'Regional Specialty & Food Enthusiasts',
            'characteristics': ['Authentic-seeking', 'Food lovers', 'Willing to travel', 'Quality-focused'],
            'peak_times': ['11:00-14:00', '18:00-21:00', 'Weekends'],
            'cuisine_types': ['South Indian', 'Bengali', 'Gujarati', 'Maharashtrian', 'Punjabi', 'Coastal regional'],
            'products': ['Authentic regional dishes', 'Special preparations', 'Traditional sweets', 'Heritage recipes'],
            'avg_spending': '₹300-₹1000',
            'payment_methods': ['Cash', 'Card', 'UPI'],
            'price_strategy': 'Premium for authenticity, heritage pricing',
            'marketing': [
                'Food blogs & review sites',
                'Facebook heritage/culture groups',
                'WhatsApp food communities',
                'Google Maps reviews',
                'Instagram storytelling (traditional methods)',
                'Food festivals & events'
            ]
        },
        'catering_events': {
            'name': 'Bulk Catering & Event Customers',
            'characteristics': ['Occasion-driven', 'Value-conscious for bulk', 'Reliability-focused', 'Planner mentality'],
            'peak_times': ['Weekends', 'Evenings for planning', 'Peak wedding/event season'],
            'cuisine_types': ['Multi-cuisine buffet', 'Customizable menus', 'Traditional & modern mix'],
            'products': ['Bulk meals', 'Customized menus', 'Setup & service', 'Themed catering'],
            'avg_spending': '₹10,000-₹100,000+',
            'payment_methods': ['Bank transfer', 'Check', 'UPI', 'Credit card'],
            'price_strategy': 'Per-plate pricing with bulk discounts, customization charges',
            'marketing': [
                'WhatsApp business links',
                'Wedding portals (ShaadiDotCom, etc)',
                'Event planner networks',
                'Facebook pages & groups',
                'Word-of-mouth referrals',
                'Portfolio/samples focus'
            ]
        },
        'health_organic': {
            'name': 'Health-Conscious & Organic Customers',
            'characteristics': ['Health-focused', 'Quality-obsessed', 'Premium willing', 'Environmentally aware'],
            'peak_times': ['07:00-09:00', '12:30-13:30', '18:00-20:00'],
            'cuisine_types': ['Organic meals', 'Vegan options', 'Protein bowls', 'Sugar-free', 'Gluten-free'],
            'products': ['Healthy meals', 'Organic produce', 'Fitness bowls', 'Detox juices', 'Supplement shakes'],
            'avg_spending': '₹300-₹800',
            'payment_methods': ['Card', 'UPI', 'App payments', 'Subscriptions'],
            'price_strategy': 'Premium for health benefits, subscription models',
            'marketing': [
                'Instagram wellness community',
                'Health & fitness influencers',
                'Fitness app partnerships',
                'Email wellness newsletters',
                'YouTube health channel ads',
                'Facebook health groups'
            ]
        }
    }

    @staticmethod
    def identify_persona(criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify food business customer persona based on provided criteria (India-specific)
        
        Criteria can include:
        - time: current time (0-23)
        - budget_level: 'low', 'medium', 'high'
        - food_type: 'street_food', 'home_delivery', 'dine_in', 'catering', 'health_conscious'
        - occasion: 'daily_meal', 'casual_outing', 'celebration', 'event'
        - customer_type: 'student', 'working_professional', 'family', 'food_enthusiast', 'event_planner'
        - delivery_preference: 'no_delivery', 'app_delivery', 'self_pickup'
        """
        
        scores = {}
        
        for persona_key, persona_data in CustomerPersona.PERSONAS.items():
            score = 0
            reasons = []
            
            if 'time' in criteria:
                current_time = criteria['time']
                peak_times = persona_data['peak_times']
                if CustomerPersona._is_peak_time(current_time, peak_times):
                    score += 25
                    reasons.append(f"Peak time match: {current_time}:00")
            
            if 'budget_level' in criteria:
                budget = criteria['budget_level']
                avg_spending = persona_data['avg_spending']
                
                if budget == 'low' and persona_key in ['street_food', 'budget_customers']:
                    score += 25
                    reasons.append("Budget-conscious customer")
                elif budget == 'medium' and persona_key in ['fast_casual', 'regional_specialty']:
                    score += 25
                    reasons.append("Medium spending power")
                elif budget == 'high' and persona_key in ['fine_dining', 'premium_users']:
                    score += 25
                    reasons.append("Premium spending capacity")
            
            if 'food_type' in criteria:
                food_type = criteria['food_type']
                if food_type == 'street_food' and persona_key == 'street_food':
                    score += 20
                    reasons.append("Street food preference matched")
                elif food_type == 'home_delivery' and persona_key == 'cloud_kitchen':
                    score += 20
                    reasons.append("Delivery app user")
                elif food_type == 'dine_in' and persona_key in ['fast_casual', 'fine_dining']:
                    score += 15
                    reasons.append("Dine-in preference")
                elif food_type == 'catering' and persona_key == 'catering_events':
                    score += 25
                    reasons.append("Event/catering requirement")
                elif food_type == 'health_conscious' and persona_key == 'health_organic':
                    score += 25
                    reasons.append("Health-conscious food seeker")
            
            if 'occasion' in criteria:
                occasion = criteria['occasion']
                if occasion == 'daily_meal' and persona_key in ['street_food', 'fast_casual', 'cloud_kitchen']:
                    score += 15
                    reasons.append("Regular daily consumption")
                elif occasion == 'casual_outing' and persona_key in ['fast_casual', 'regional_specialty']:
                    score += 15
                    reasons.append("Casual dining occasion")
                elif occasion == 'celebration' and persona_key in ['fine_dining', 'catering_events']:
                    score += 20
                    reasons.append("Special occasion detected")
                elif occasion == 'event' and persona_key == 'catering_events':
                    score += 30
                    reasons.append("Bulk event requirement")
            
            if 'customer_type' in criteria:
                ctype = criteria['customer_type']
                if ctype == 'student' and persona_key == 'street_food':
                    score += 20
                    reasons.append("Student customer profile")
                elif ctype == 'working_professional' and persona_key in ['cloud_kitchen', 'fast_casual']:
                    score += 20
                    reasons.append("Working professional - delivery preference")
                elif ctype == 'family' and persona_key == 'fast_casual':
                    score += 20
                    reasons.append("Family dining preference")
                elif ctype == 'food_enthusiast' and persona_key == 'regional_specialty':
                    score += 20
                    reasons.append("Food enthusiast/connoisseur")
                elif ctype == 'event_planner' and persona_key == 'catering_events':
                    score += 25
                    reasons.append("Event planner/organizer")
            
            if 'delivery_preference' in criteria:
                delivery = criteria['delivery_preference']
                if delivery == 'app_delivery' and persona_key == 'cloud_kitchen':
                    score += 20
                    reasons.append("Zomato/Swiggy user")
                elif delivery == 'self_pickup' and persona_key in ['street_food', 'regional_specialty']:
                    score += 15
                    reasons.append("Direct pickup preference")
                elif delivery == 'no_delivery' and persona_key in ['fine_dining', 'catering_events']:
                    score += 15
                    reasons.append("In-person/direct engagement")
            
            if 'payment_method' in criteria:
                payment = criteria['payment_method']
                if payment in ['upi', 'paytm', 'cash'] and persona_key in ['street_food']:
                    score += 10
                    reasons.append("Digital/cash payment ready")
                elif payment in ['card', 'online_transfer'] and persona_key in ['fine_dining', 'catering_events']:
                    score += 10
                    reasons.append("Formal payment method")
            
            if score > 0:
                scores[persona_key] = {'score': score, 'reasons': reasons}
        
        if not scores:
            return {
                'persona_key': 'general',
                'persona_name': 'General Customer',
                'confidence': 0,
                'message': 'Insufficient data for persona identification'
            }
        
        best_persona = max(scores.items(), key=lambda x: x[1]['score'])
        persona_key = best_persona[0]
        persona_data = CustomerPersona.PERSONAS[persona_key]
        
        return {
            'persona_key': persona_key,
            'persona_name': persona_data['name'],
            'confidence': min(100, best_persona[1]['score']),
            'characteristics': persona_data['characteristics'],
            'cuisine_types': persona_data.get('cuisine_types', []),
            'peak_times': persona_data['peak_times'],
            'products': persona_data['products'],
            'avg_spending': persona_data['avg_spending'],
            'payment_methods': persona_data.get('payment_methods', []),
            'price_strategy': persona_data['price_strategy'],
            'marketing_strategies': persona_data['marketing'],
            'detection_reasons': best_persona[1]['reasons']
        }
    
    @staticmethod
    def _is_peak_time(current_time: int, peak_times: List[str]) -> bool:
        """Check if current time matches any peak time range"""
        for peak_range in peak_times:
            if '-' in peak_range:
                start, end = peak_range.split('-')
                start_hour = int(start.split(':')[0])
                end_hour = int(end.split(':')[0])
                if start_hour <= current_time <= end_hour:
                    return True
            elif peak_range == 'Variable - flexible timing':
                return True
        return False
    
    @staticmethod
    def get_all_personas() -> Dict[str, Any]:
        """Get all available personas"""
        return {
            key: {
                'name': data['name'],
                'characteristics': data['characteristics'],
                'avg_spending': data['avg_spending']
            }
            for key, data in CustomerPersona.PERSONAS.items()
        }
