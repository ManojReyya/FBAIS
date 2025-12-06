import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class PersonaModelTrainer:
    """Train and save persona classification model"""
    
    def __init__(self, data_path='synthetic_training_data.csv'):
        self.data_path = data_path
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        self.class_labels = None
        
    def load_data(self):
        """Load synthetic training data"""
        print("[*] Loading synthetic training data...")
        df = pd.read_csv(self.data_path)
        print(f"[+] Loaded {len(df)} records")
        return df
    
    def preprocess_data(self, df):
        """Encode categorical variables"""
        print("[*] Preprocessing data...")
        
        df_encoded = df.copy()
        
        # Columns to encode
        categorical_cols = ['budget_level', 'food_type', 'occasion', 
                           'customer_type', 'delivery_preference', 'payment_method']
        
        # Encode features
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                print(f"  [-] Encoded '{col}': {len(le.classes_)} classes")
        
        # Encode target
        target_le = LabelEncoder()
        df_encoded['persona'] = target_le.fit_transform(df_encoded['persona'])
        self.label_encoders['persona'] = target_le
        self.class_labels = target_le.classes_
        print(f"  [-] Encoded 'persona': {len(target_le.classes_)} classes")
        
        self.feature_names = ['time', 'budget_level', 'food_type', 'occasion',
                             'customer_type', 'delivery_preference', 'payment_method']
        
        return df_encoded
    
    def split_data(self, df_encoded, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"[*] Splitting data: {100-int(test_size*100)}% train, {int(test_size*100)}% test")
        
        X = df_encoded[self.feature_names]
        y = df_encoded['persona']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"[+] Train set: {len(X_train)} samples")
        print(f"[+] Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("[*] Training Random Forest classifier...")
        print("  [-] Initializing model with 200 trees...")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        print("[+] Model training completed!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n[*] Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n[+] Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"[+] F1 Score (weighted): {f1:.4f}")
        
        print("\n[+] Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.class_labels, 
                                   digits=4))
        
        # Feature importance
        print("\n[+] Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.to_string(index=False))
        
        return accuracy, f1
    
    def save_model(self, model_path='persona_classifier.pkl', 
                   encoders_path='label_encoders.pkl',
                   features_path='feature_names.pkl',
                   metadata_path='model_metadata.json'):
        """Save model and encoders"""
        print(f"\n[*] Saving model artifacts...")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[+] Model saved to: {model_path}")
        
        # Save label encoders
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"[+] Encoders saved to: {encoders_path}")
        
        # Save feature names
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"[+] Feature names saved to: {features_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'classes': list(self.class_labels),
            'n_classes': len(self.class_labels),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_samples': 10500,
            'model_version': '1.0'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[+] Metadata saved to: {metadata_path}")
    
    def train_and_save(self):
        """Complete training pipeline"""
        print("=" * 60)
        print("[*] Persona Classification Model Training Pipeline")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Preprocess
        df_encoded = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_encoded)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate
        accuracy, f1 = self.evaluate_model(X_test, y_test)
        
        # Save
        self.save_model()
        
        print("\n" + "=" * 60)
        print("[+] Training completed successfully!")
        print("=" * 60)
        
        return accuracy, f1


def main():
    trainer = PersonaModelTrainer(data_path='synthetic_training_data.csv')
    accuracy, f1 = trainer.train_and_save()


if __name__ == '__main__':
    main()
