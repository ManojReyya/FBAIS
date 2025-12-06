import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("[*] Testing /api/model-info endpoint")
    print("="*60)
    
    try:
        response = requests.get(f'{BASE_URL}/api/model-info')
        result = response.json()
        print(json.dumps(result, indent=2))
        return True
    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def test_persona_prediction(test_case):
    """Test persona prediction"""
    print(f"\n[*] Test: {test_case['name']}")
    print("-" * 60)
    
    try:
        response = requests.post(
            f'{BASE_URL}/api/identify',
            json=test_case['criteria'],
            headers={'Content-Type': 'application/json'}
        )
        result = response.json()
        
        print(f"  Predicted Persona: {result.get('persona_name', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0):.1f}%")
        print(f"  Model Used: {result.get('model_used', 'N/A')}")
        
        if result.get('all_predictions'):
            top_3 = sorted(result['all_predictions'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3 Predictions:")
            for persona, prob in top_3:
                print(f"    - {persona}: {prob*100:.1f}%")
        
        print(f"  Detection Reasons:")
        for reason in result.get('detection_reasons', []):
            print(f"    - {reason}")
        
        return True
    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def run_all_tests():
    """Run all test cases"""
    
    test_cases = [
        {
            'name': 'Student - Street Food (Low Budget)',
            'criteria': {
                'time': 8,
                'budget_level': 'low',
                'food_type': 'street_food',
                'occasion': 'daily_meal',
                'customer_type': 'student',
                'delivery_preference': 'self_pickup',
                'payment_method': 'cash'
            }
        },
        {
            'name': 'Working Professional - Cloud Kitchen (Medium Budget)',
            'criteria': {
                'time': 12,
                'budget_level': 'medium',
                'food_type': 'cloud_kitchen',
                'occasion': 'daily_meal',
                'customer_type': 'working_professional',
                'delivery_preference': 'app_delivery',
                'payment_method': 'upi'
            }
        },
        {
            'name': 'Food Enthusiast - Regional Specialty',
            'criteria': {
                'time': 19,
                'budget_level': 'high',
                'food_type': 'regional_specialty',
                'occasion': 'casual_outing',
                'customer_type': 'food_enthusiast',
                'delivery_preference': 'self_pickup',
                'payment_method': 'card'
            }
        },
        {
            'name': 'Family - Fast Casual (Medium Budget)',
            'criteria': {
                'time': 13,
                'budget_level': 'medium',
                'food_type': 'fast_casual',
                'occasion': 'casual_outing',
                'customer_type': 'family',
                'delivery_preference': 'no_delivery',
                'payment_method': 'card'
            }
        },
        {
            'name': 'Event Planner - Catering Events (High Budget)',
            'criteria': {
                'time': 18,
                'budget_level': 'high',
                'food_type': 'catering_events',
                'occasion': 'event',
                'customer_type': 'event_planner',
                'delivery_preference': 'no_delivery',
                'payment_method': 'online_transfer'
            }
        },
        {
            'name': 'Health-Conscious - Organic Meals',
            'criteria': {
                'time': 7,
                'budget_level': 'medium',
                'food_type': 'health_organic',
                'occasion': 'daily_meal',
                'customer_type': 'working_professional',
                'delivery_preference': 'app_delivery',
                'payment_method': 'upi'
            }
        },
        {
            'name': 'Fine Dining - Celebration',
            'criteria': {
                'time': 20,
                'budget_level': 'high',
                'food_type': 'fine_dining',
                'occasion': 'celebration',
                'customer_type': 'food_enthusiast',
                'delivery_preference': 'no_delivery',
                'payment_method': 'card'
            }
        }
    ]
    
    print("\n" + "="*60)
    print("[*] ML Persona Prediction Tests")
    print("="*60)
    
    # Test model info
    test_model_info()
    
    # Test predictions
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        if test_persona_prediction(test_case):
            passed += 1
        else:
            failed += 1
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("[+] Test Results")
    print("="*60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {passed + failed}")


if __name__ == '__main__':
    print("[*] Make sure Flask app is running on http://localhost:5000")
    print("[*] Run: python app.py")
    input("Press Enter to start tests...")
    
    run_all_tests()
