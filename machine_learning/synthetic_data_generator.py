import random
import numpy as np
import pandas as pd
from datetime import datetime
from personas import CustomerPersona

class RealisticSyntheticDataGenerator:
    """
    Generate realistic synthetic customer data for Indian food business
    with proper statistical distributions and cross-correlations
    """
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def generate_time(self, persona_key):
        """
        Generate realistic time with normal distribution around peak hours
        """
        peak_times = CustomerPersona.PERSONAS[persona_key]['peak_times']
        
        # Find first valid peak time (with hour:minute format)
        peak_hour = 12  # Default
        for peak_time in peak_times:
            if ':' in peak_time and peak_time not in ['Weekends', 'Variable - flexible timing']:
                try:
                    peak_hour = int(peak_time.split('-')[0].split(':')[0])
                    break
                except:
                    continue
        
        # Add some randomness with normal distribution (σ=1.5 hours)
        hour = int(np.random.normal(peak_hour, 1.5))
        hour = max(0, min(23, hour))  # Clamp to 0-23
        
        return hour
    
    def get_budget_for_persona(self, persona_key):
        """
        Get realistic budget distribution for each persona
        """
        distributions = {
            'street_food': {
                'low': 0.70,      # 70% low budget
                'medium': 0.25,   # 25% medium
                'high': 0.05      # 5% occasional high
            },
            'fast_casual': {
                'low': 0.15,
                'medium': 0.70,   # 70% medium
                'high': 0.15
            },
            'fine_dining': {
                'low': 0.02,
                'medium': 0.18,
                'high': 0.80      # 80% high budget
            },
            'cloud_kitchen': {
                'low': 0.20,
                'medium': 0.65,   # 65% medium
                'high': 0.15
            },
            'regional_specialty': {
                'low': 0.10,
                'medium': 0.60,
                'high': 0.30
            },
            'catering_events': {
                'low': 0.02,
                'medium': 0.28,
                'high': 0.70      # 70% high budget
            },
            'health_organic': {
                'low': 0.05,
                'medium': 0.50,
                'high': 0.45      # 45% premium willing
            }
        }
        
        dist = distributions.get(persona_key, {'low': 0.33, 'medium': 0.34, 'high': 0.33})
        return np.random.choice(['low', 'medium', 'high'], p=[dist['low'], dist['medium'], dist['high']])
    
    def get_occasion_for_budget_and_type(self, budget, persona_key):
        """
        Realistic occasion based on budget and persona
        """
        if budget == 'low':
            return np.random.choice(
                ['daily_meal', 'casual_outing', 'celebration'],
                p=[0.70, 0.25, 0.05]
            )
        elif budget == 'high':
            return np.random.choice(
                ['daily_meal', 'casual_outing', 'celebration', 'event'],
                p=[0.20, 0.30, 0.30, 0.20]
            )
        else:  # medium
            return np.random.choice(
                ['daily_meal', 'casual_outing', 'celebration'],
                p=[0.50, 0.35, 0.15]
            )
    
    def get_customer_type_for_persona(self, persona_key):
        """
        Realistic customer type distribution per persona
        """
        mapping = {
            'street_food': {
                'student': 0.40,
                'working_professional': 0.35,
                'family': 0.15,
                'food_enthusiast': 0.08,
                'event_planner': 0.02
            },
            'fast_casual': {
                'student': 0.20,
                'working_professional': 0.30,
                'family': 0.40,
                'food_enthusiast': 0.08,
                'event_planner': 0.02
            },
            'fine_dining': {
                'student': 0.05,
                'working_professional': 0.30,
                'family': 0.20,
                'food_enthusiast': 0.40,
                'event_planner': 0.05
            },
            'cloud_kitchen': {
                'student': 0.25,
                'working_professional': 0.60,
                'family': 0.10,
                'food_enthusiast': 0.04,
                'event_planner': 0.01
            },
            'regional_specialty': {
                'student': 0.10,
                'working_professional': 0.25,
                'family': 0.20,
                'food_enthusiast': 0.40,
                'event_planner': 0.05
            },
            'catering_events': {
                'student': 0.02,
                'working_professional': 0.15,
                'family': 0.20,
                'food_enthusiast': 0.13,
                'event_planner': 0.50
            },
            'health_organic': {
                'student': 0.15,
                'working_professional': 0.50,
                'family': 0.20,
                'food_enthusiast': 0.12,
                'event_planner': 0.03
            }
        }
        
        dist = mapping.get(persona_key, {
            'student': 0.20,
            'working_professional': 0.30,
            'family': 0.25,
            'food_enthusiast': 0.20,
            'event_planner': 0.05
        })
        
        choices = list(dist.keys())
        probs = list(dist.values())
        return np.random.choice(choices, p=probs)
    
    def get_delivery_preference_for_type(self, customer_type, persona_key):
        """
        Realistic delivery preference based on customer type
        """
        if customer_type == 'working_professional':
            return np.random.choice(
                ['app_delivery', 'self_pickup', 'no_delivery'],
                p=[0.70, 0.20, 0.10]
            )
        elif customer_type == 'student':
            return np.random.choice(
                ['app_delivery', 'self_pickup', 'no_delivery'],
                p=[0.60, 0.35, 0.05]
            )
        elif customer_type == 'family':
            return np.random.choice(
                ['self_pickup', 'no_delivery', 'app_delivery'],
                p=[0.40, 0.50, 0.10]
            )
        elif customer_type == 'food_enthusiast':
            return np.random.choice(
                ['no_delivery', 'self_pickup', 'app_delivery'],
                p=[0.50, 0.40, 0.10]
            )
        else:  # event_planner
            return np.random.choice(
                ['no_delivery', 'self_pickup', 'app_delivery'],
                p=[0.80, 0.15, 0.05]
            )
    
    def get_payment_method_for_budget(self, budget, customer_type):
        """
        Realistic payment method based on budget and customer type
        """
        if budget == 'low':
            return np.random.choice(
                ['cash', 'upi', 'paytm', 'card'],
                p=[0.60, 0.30, 0.07, 0.03]
            )
        elif budget == 'high':
            return np.random.choice(
                ['card', 'upi', 'online_transfer', 'cash'],
                p=[0.50, 0.35, 0.10, 0.05]
            )
        else:  # medium
            return np.random.choice(
                ['upi', 'cash', 'card', 'paytm'],
                p=[0.40, 0.35, 0.20, 0.05]
            )
    
    def generate_sample(self, persona_key):
        """Generate single realistic sample for persona"""
        
        time = self.generate_time(persona_key)
        budget = self.get_budget_for_persona(persona_key)
        occasion = self.get_occasion_for_budget_and_type(budget, persona_key)
        customer_type = self.get_customer_type_for_persona(persona_key)
        delivery_pref = self.get_delivery_preference_for_type(customer_type, persona_key)
        payment = self.get_payment_method_for_budget(budget, customer_type)
        
        return {
            'time': time,
            'budget_level': budget,
            'food_type': persona_key,
            'occasion': occasion,
            'customer_type': customer_type,
            'delivery_preference': delivery_pref,
            'payment_method': payment,
            'persona': persona_key
        }
    
    def generate_dataset(self, samples_per_persona=1500):
        """
        Generate complete synthetic dataset
        Default: 1500 samples × 7 personas = 10,500 total samples
        """
        all_samples = []
        personas = list(CustomerPersona.PERSONAS.keys())
        
        print(f"[*] Generating synthetic data...")
        print(f"[*] Samples per persona: {samples_per_persona}")
        print(f"[*] Total personas: {len(personas)}")
        print(f"[*] Total samples: {samples_per_persona * len(personas)}\n")
        
        for persona_key in personas:
            print(f"  [-] Generating {samples_per_persona} samples for '{persona_key}'...")
            for _ in range(samples_per_persona):
                sample = self.generate_sample(persona_key)
                all_samples.append(sample)
        
        df = pd.DataFrame(all_samples)
        
        print(f"\n[+] Dataset generated successfully!")
        print(f"[+] Shape: {df.shape}")
        print(f"\n[+] Distribution by Persona:")
        print(df['persona'].value_counts())
        
        return df


def create_and_save_dataset(output_path='synthetic_training_data.csv', samples_per_persona=1500):
    """
    Create synthetic dataset and save to CSV
    """
    generator = RealisticSyntheticDataGenerator(random_seed=42)
    df = generator.generate_dataset(samples_per_persona)
    
    df.to_csv(output_path, index=False)
    print(f"\n[+] Dataset saved to: {output_path}")
    
    # Print statistics
    print(f"\n[+] Dataset Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Budget distribution:\n{df['budget_level'].value_counts()}")
    print(f"\n  Customer type distribution:\n{df['customer_type'].value_counts()}")
    print(f"\n  Payment method distribution:\n{df['payment_method'].value_counts()}")
    
    return df


if __name__ == '__main__':
    # Generate dataset
    df = create_and_save_dataset()
    
    # Show sample records
    print("\n[+] Sample Records (first 10):")
    print(df.head(10).to_string())
