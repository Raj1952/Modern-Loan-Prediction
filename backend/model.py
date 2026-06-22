import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

class LoanSimulatorModel:
    def __init__(self, data_path):
        self.data_path = data_path
        # Limit depth to keep rules human-readable
        self.model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def train(self):
        print(f"Training on {self.data_path}")
        df = pd.read_csv(self.data_path)
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Drop ID
        if 'loan_id' in df.columns:
            df = df.drop('loan_id', axis=1)
            
        # Clean string values
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            
        target_col = 'loan_status'
        X = df.drop(target_col, axis=1)
        y = df[target_col].apply(lambda x: 1 if x == 'Approved' else 0)
        
        self.feature_names = X.columns.tolist()
        
        # Encode categorical
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            
        self.model.fit(X, y)
        self.is_trained = True
        acc = self.model.score(X, y)
        
        # We can extract the global rules as text for the admin
        tree_rules = export_text(self.model, feature_names=self.feature_names)
        
        return {"status": "Model trained successfully", "accuracy": acc, "rules": tree_rules}
        
    def predict_explainable(self, input_data):
        if not self.is_trained:
            self.train()
            
        input_df = pd.DataFrame([input_data])
        
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
                
            if col in self.label_encoders:
                val = str(input_df[col].iloc[0]).strip()
                le = self.label_encoders[col]
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = 0 # Fallback for unknown
                    
        X_pred = input_df[self.feature_names]
        
        prediction = self.model.predict(X_pred)[0]
        status = "Approved" if prediction == 1 else "Rejected"
        
        # Extract path
        node_indicator = self.model.decision_path(X_pred)
        leaf_id = self.model.apply(X_pred)[0]
        
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        reasons = []
        for node_id in node_index:
            if leaf_id == node_id:
                continue
                
            feat_name = self.feature_names[feature[node_id]]
            feat_val = X_pred.iloc[0, feature[node_id]]
            thresh_val = threshold[node_id]
            
            # Map categorical back to readable names if possible
            display_val = feat_val
            if feat_name in self.label_encoders:
                # the encoded value
                encoded_classes = self.label_encoders[feat_name].classes_
                if int(feat_val) < len(encoded_classes):
                    display_val = encoded_classes[int(feat_val)]
                    
            if (X_pred.iloc[0, feature[node_id]] <= threshold[node_id]):
                reasons.append(f"{feat_name} ({display_val}) is <= {thresh_val:.1f}")
            else:
                reasons.append(f"{feat_name} ({display_val}) is > {thresh_val:.1f}")
                
        reason_str = " AND ".join(reasons)
        return {
            "status": status,
            "decision_path": reasons,
            "message": f"Application {status}. Reason: {reason_str}"
        }
