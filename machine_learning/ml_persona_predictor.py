import pickle
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'features'))
from personas import CustomerPersona

class MLPersonaPredictor:
    """
    Use trained ML model to predict customer personas
    Hybrid approach: ML model with rule-based fallback
    """
    
    def __init__(self, model_path=None,
                 encoders_path=None,
                 features_path=None,
                 metadata_path=None):
        
        base_dir = os.path.dirname(__file__)
        model_path = model_path or os.path.join(base_dir, 'persona_classifier.pkl')
        encoders_path = encoders_path or os.path.join(base_dir, 'label_encoders.pkl')
        features_path = features_path or os.path.join(base_dir, 'feature_names.pkl')
        metadata_path = metadata_path or os.path.join(base_dir, 'model_metadata.json')
        
        print("[*] Loading ML model artifacts...")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[+] Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"[-] Model not found at {model_path}")
            self.model = None
        
        try:
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print(f"[+] Encoders loaded from {encoders_path}")
        except FileNotFoundError:
            print(f"[-] Encoders not found at {encoders_path}")
            self.label_encoders = None
        
        try:
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"[+] Feature names loaded from {features_path}")
        except FileNotFoundError:
            print(f"[-] Features not found at {features_path}")
            self.feature_names = None
        
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"[+] Metadata loaded from {metadata_path}")
        except FileNotFoundError:
            print(f"[-] Metadata not found at {metadata_path}")
            self.metadata = None
        
        self.model_available = (self.model is not None and 
                               self.label_encoders is not None and 
                               self.feature_names is not None)
    
    def prepare_features(self, criteria):
        """
        Prepare and encode features for ML model prediction
        """
        if not self.model_available:
            return None
        
        try:
            # Map criteria to feature names and encode
            features_dict = {}
            
            for feature in self.feature_names:
                if feature == 'time':
                    features_dict[feature] = criteria.get('time', 12)
                elif feature in criteria:
                    # Encode categorical feature
                    value = criteria[feature]
                    encoder = self.label_encoders.get(feature)
                    
                    if encoder:
                        try:
                            encoded_value = encoder.transform([value])[0]
                            features_dict[feature] = encoded_value
                        except ValueError:
                            # Value not in training set, use default
                            print(f"  [-] Value '{value}' not in encoder for '{feature}', using default")
                            features_dict[feature] = 0
                    else:
                        features_dict[feature] = 0
                else:
                    features_dict[feature] = 0
            
            # Create feature array in correct order
            X = np.array([[features_dict[f] for f in self.feature_names]])
            return X
        
        except Exception as e:
            print(f"[-] Error preparing features: {e}")
            return None
    
    def predict_ml(self, criteria):
        """
        Predict persona using ML model
        Returns: persona_key, confidence, probabilities
        """
        if not self.model_available:
            return None, 0, None
        
        try:
            X = self.prepare_features(criteria)
            if X is None:
                return None, 0, None
            
            # Get prediction and probabilities
            persona_encoded = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Decode persona
            persona_encoder = self.label_encoders.get('persona')
            if persona_encoder:
                persona_key = persona_encoder.inverse_transform([persona_encoded])[0]
            else:
                return None, 0, None
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Get all predictions with probabilities
            persona_probs = {
                persona_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return persona_key, confidence, persona_probs
        
        except Exception as e:
            print(f"[-] Error in ML prediction: {e}")
            return None, 0, None
    
    def predict_hybrid(self, criteria, ml_threshold=0.70):
        """
        Hybrid prediction: Use ML model if confident, fallback to rule-based
        
        Args:
            criteria: Customer criteria dict
            ml_threshold: Minimum confidence to trust ML model
        
        Returns:
            Full prediction result with metadata
        """
        
        # Get ML prediction
        ml_persona, ml_confidence, ml_probs = self.predict_ml(criteria)
        
        # Get rule-based prediction
        rule_result = CustomerPersona.identify_persona(criteria)
        
        # Decide which to use
        if ml_persona and ml_confidence >= ml_threshold:
            # Use ML prediction
            result = {
                'persona_key': ml_persona,
                'persona_name': CustomerPersona.PERSONAS[ml_persona]['name'],
                'confidence': min(100, ml_confidence * 100),
                'characteristics': CustomerPersona.PERSONAS[ml_persona]['characteristics'],
                'cuisine_types': CustomerPersona.PERSONAS[ml_persona].get('cuisine_types', []),
                'peak_times': CustomerPersona.PERSONAS[ml_persona]['peak_times'],
                'products': CustomerPersona.PERSONAS[ml_persona]['products'],
                'avg_spending': CustomerPersona.PERSONAS[ml_persona]['avg_spending'],
                'payment_methods': CustomerPersona.PERSONAS[ml_persona].get('payment_methods', []),
                'price_strategy': CustomerPersona.PERSONAS[ml_persona]['price_strategy'],
                'marketing_strategies': CustomerPersona.PERSONAS[ml_persona]['marketing'],
                'detection_reasons': [f"ML Model (confidence: {ml_confidence*100:.1f}%)"],
                'model_used': 'ML_MODEL',
                'all_predictions': ml_probs
            }
        else:
            # Use rule-based (fallback)
            result = rule_result.copy()
            result['model_used'] = 'RULE_BASED'
            result['all_predictions'] = ml_probs if ml_probs else None
        
        return result
    
    def explain_prediction(self, criteria):
        """
        Explain why a specific prediction was made
        """
        ml_persona, ml_confidence, ml_probs = self.predict_ml(criteria)
        rule_result = CustomerPersona.identify_persona(criteria)
        
        explanation = {
            'ml_prediction': {
                'persona': ml_persona,
                'confidence': ml_confidence,
                'top_3_predictions': sorted(ml_probs.items(), key=lambda x: x[1], reverse=True)[:3] if ml_probs else []
            },
            'rule_based_prediction': {
                'persona': rule_result.get('persona_key'),
                'confidence': rule_result.get('confidence'),
                'reasons': rule_result.get('detection_reasons', [])
            },
            'model_status': 'AVAILABLE' if self.model_available else 'NOT_AVAILABLE'
        }
        
        return explanation


def test_ml_predictor():
    """Test the ML predictor"""
    print("\n" + "="*60)
    print("[*] Testing ML Persona Predictor")
    print("="*60 + "\n")
    
    # Initialize predictor
    predictor = MLPersonaPredictor()
    
    if not predictor.model_available:
        print("[-] ML model not available, only rule-based predictions will work")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'Student (Street Food)',
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
            'name': 'Working Professional (Cloud Kitchen)',
            'criteria': {
                'time': 12,
                'budget_level': 'medium',
                'food_type': 'home_delivery',
                'occasion': 'daily_meal',
                'customer_type': 'working_professional',
                'delivery_preference': 'app_delivery',
                'payment_method': 'upi'
            }
        },
        {
            'name': 'Food Enthusiast (Fine Dining)',
            'criteria': {
                'time': 20,
                'budget_level': 'high',
                'food_type': 'dine_in',
                'occasion': 'celebration',
                'customer_type': 'food_enthusiast',
                'delivery_preference': 'no_delivery',
                'payment_method': 'card'
            }
        },
        {
            'name': 'Event Planner (Catering)',
            'criteria': {
                'time': 18,
                'budget_level': 'high',
                'food_type': 'catering',
                'occasion': 'event',
                'customer_type': 'event_planner',
                'delivery_preference': 'no_delivery',
                'payment_method': 'online_transfer'
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"[*] Test: {test_case['name']}")
        print("-" * 60)
        
        result = predictor.predict_hybrid(test_case['criteria'])
        
        print(f"  Predicted Persona: {result['persona_name']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Model Used: {result['model_used']}")
        
        if result.get('all_predictions'):
            print(f"  All Predictions:")
            for persona, prob in sorted(result['all_predictions'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"    - {persona}: {prob*100:.1f}%")
        
        print()


if __name__ == '__main__':
    test_ml_predictor()
