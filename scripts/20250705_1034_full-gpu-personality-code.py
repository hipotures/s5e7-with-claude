import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.integration import CatBoostPruningCallback
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Sprawdzenie GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

class DeepPersonalityAnalyzer:
    """
    GPU-accelerated personality analysis with neural networks and Optuna optimization
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.best_params = {}
        self.models = {}
        
    def prepare_data(self, train_path, test_path):
        """Przygotowanie danych z zaawansowanym feature engineering"""
        print("📊 Loading and preparing data...")
        print(f"  Reading train data from: {train_path}")
        print(f"  Reading test data from: {test_path}")
        
        # Sprawdzenie czy pliki istnieją
        import os
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Wczytanie plików CSV
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"  ✓ Train data loaded: {train_df.shape}")
        print(f"  ✓ Test data loaded: {test_df.shape}")
        
        # Feature engineering
        for df in [train_df, test_df]:
            # Wskaźniki braków
            for col in ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                       'Post_frequency']:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
            
            # Konwersja Yes/No
            for col in ['Stage_fear', 'Drained_after_socializing']:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
            
            # Imputacja
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'id' in numeric_cols:
                numeric_cols = numeric_cols.drop('id')
            if 'Personality' in df.columns and 'Personality' in numeric_cols:
                numeric_cols = numeric_cols.drop('Personality')
            
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # Zaawansowane cechy
            df['social_score'] = (df['Social_event_attendance'] + 
                                 (10 - df['Time_spent_Alone']) + 
                                 df['Friends_circle_size'] / 3) / 3
            
            df['anxiety_score'] = (df['Stage_fear'] + 
                                  (10 - df['Going_outside']) +
                                  df['Drained_after_socializing']) / 3
            
            df['digital_social_ratio'] = (df['Post_frequency'] + 1) / (df['Social_event_attendance'] + 1)
            df['total_missing'] = df[[f'{col}_missing' for col in ['Time_spent_Alone', 'Stage_fear', 
                                     'Social_event_attendance', 'Going_outside', 
                                     'Drained_after_socializing', 'Friends_circle_size',
                                     'Post_frequency']]].sum(axis=1)
            
            # Polynomial features dla najważniejszych zmiennych
            df['social_anxiety_interaction'] = df['social_score'] * df['anxiety_score']
            df['social_squared'] = df['social_score'] ** 2
            df['anxiety_squared'] = df['anxiety_score'] ** 2
        
        return train_df, test_df
    
    def generate_synthetic_ambiverts(self, X, y, target_ratio=0.15):
        """Generowanie syntetycznych ambiwertyków używając SMOTE"""
        print(f"\n🧬 Generating synthetic ambiverts (target ratio: {target_ratio})...")
        
        # Najpierw znajdujemy potencjalnych ambiwertyków
        # Trenujemy prosty model do uzyskania prawdopodobieństw
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        probs = rf.predict_proba(X)[:, 1]
        
        # Identyfikujemy "graniczne" przypadki
        ambivert_mask = (probs > 0.35) & (probs < 0.65)
        ambivert_indices = np.where(ambivert_mask)[0]
        
        # Tworzymy nową klasę dla ambiwertyków
        y_3class = y.copy()
        y_3class[ambivert_indices] = 2  # 0=Intro, 1=Extro, 2=Ambi
        
        # Używamy SMOTE do wygenerowania więcej ambiwertyków
        # Określamy które kolumny są kategoryczne
        categorical_features = [i for i, col in enumerate(X.columns) if 'missing' in col]
        
        smote = SMOTENC(
            categorical_features=categorical_features,
            sampling_strategy={2: int(len(X) * target_ratio)},
            random_state=42,
            k_neighbors=5
        )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y_3class)
            print(f"Generated {sum(y_resampled == 2)} ambiverts (total samples: {len(y_resampled)})")
            
            # Konwertujemy z powrotem do 2 klas dla podstawowego treningu
            y_2class = (y_resampled != 0).astype(int)  # 0=Intro, 1,2=Extro+Ambi
            
            return X_resampled, y_2class, y_resampled
        except:
            print("SMOTE failed, returning original data")
            return X, y, y_3class

class PersonalityAutoencoder(nn.Module):
    """Autoencoder do nauki reprezentacji osobowości"""
    
    def __init__(self, input_dim, encoding_dim=16, dropout=0.2):
        super(PersonalityAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class PersonalityClassifier(nn.Module):
    """Neural network classifier dla 3 klas osobowości"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(PersonalityClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))  # 3 klasy
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def objective_xgboost(trial, X, y, cv_folds=5):
    """Optuna objective dla XGBoost 3.x z GPU
    
    Zgodne z https://xgboost.readthedocs.io/en/stable/python/index.html
    """
    
    # XGBoost 3.x parameters
    params = {
        # Basic parameters
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        
        # GPU parameters (XGBoost 3.x)
        'device': 'cuda:0',  # Nowy sposób specyfikacji GPU w XGBoost 3.x
        'tree_method': 'hist',  # 'gpu_hist' jest deprecated, używamy 'hist' z device='cuda'
        
        # Hyperparameters do optymalizacji
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        
        # Early stopping w konstruktorze (XGBoost 3.x)
        'early_stopping_rounds': 10,
        
        # Additional parameters
        'random_state': 42,
        'verbosity': 0  # Zamiast verbose=False
    }
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # XGBoost 3.x approach
        model = xgb.XGBClassifier(**params)
        
        # Fit with early stopping (XGBoost 3.x style)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get best iteration score for pruning
        if hasattr(model, 'best_score'):
            trial.report(model.best_score, fold)
            
        # Prune if necessary
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        scores.append(score)
    
    return np.mean(scores)

def objective_catboost(trial, X, y, cv_folds=5):
    """Optuna objective dla CatBoost z GPU"""
    
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'task_type': 'GPU',
        'devices': '0:1',  # Używa obu GPU
        'loss_function': 'MultiClass',
        'verbose': False,
        'random_state': 42
    }
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        scores.append(score)
    
    return np.mean(scores)

def train_neural_ensemble(X_train, y_train, X_val, y_val, device='cuda'):
    """Trenowanie ensemble modeli neuronowych"""
    
    # Konwersja do tensorów
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Dataset i DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    models = []
    
    # Trenujemy kilka modeli z różnymi architekturami
    architectures = [
        [128, 64, 32],
        [256, 128, 64, 32],
        [128, 128, 64],
        [256, 64, 32]
    ]
    
    for i, hidden_dims in enumerate(architectures):
        print(f"\nTraining neural model {i+1}/{len(architectures)}...")
        
        model = PersonalityClassifier(X_train.shape[1], hidden_dims).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_score = 0
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_score = f1_score(y_val, val_preds, average='macro')
            
            scheduler.step(val_loss)
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        models.append(model)
        print(f"Model {i+1} best F1: {best_val_score:.4f}")
    
    return models

def create_3class_labels(df, X, method='neural'):
    """Tworzenie 3 klas używając neural network lub ensemble"""
    print(f"\n🎯 Creating 3 classes using {method} method...")
    
    y = (df['Personality'] == 'Extrovert').astype(int)
    
    if method == 'neural':
        # Używamy autoencoder do nauki reprezentacji
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalizacja
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Trenowanie autoencoder
        autoencoder = PersonalityAutoencoder(X.shape[1], encoding_dim=16).to(device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print("Training autoencoder...")
        for epoch in range(50):
            optimizer.zero_grad()
            decoded, encoded = autoencoder(X_tensor)
            loss = criterion(decoded, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Uzyskanie reprezentacji
        with torch.no_grad():
            _, embeddings = autoencoder(X_tensor)
            embeddings = embeddings.cpu().numpy()
        
        # Clustering na embeddingach
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Mapowanie klastrów do klas osobowości
        cluster_means = []
        for c in range(3):
            mask = clusters == c
            mean_extrovert = y[mask].mean()
            cluster_means.append((c, mean_extrovert))
        
        cluster_means.sort(key=lambda x: x[1])
        
        # Przypisanie etykiet
        label_map = {
            cluster_means[0][0]: 'Strong_Introvert',
            cluster_means[1][0]: 'Ambivert',
            cluster_means[2][0]: 'Strong_Extrovert'
        }
        
        df['personality_3class'] = [label_map[c] for c in clusters]
        
    else:  # ensemble method
        # Używamy ensemble różnych modeli
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        models = [
            RandomForestClassifier(n_estimators=200, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            LogisticRegression(max_iter=1000, random_state=42)
        ]
        
        probabilities = []
        for model in models:
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
            probabilities.append(probs)
        
        # Średnia prawdopodobieństw
        avg_probs = np.mean(probabilities, axis=0)
        
        # Dynamiczne progi
        p25 = np.percentile(avg_probs, 25)
        p75 = np.percentile(avg_probs, 75)
        
        personality_3class = []
        for prob in avg_probs:
            if prob < p25:
                personality_3class.append('Strong_Introvert')
            elif prob > p75:
                personality_3class.append('Strong_Extrovert')
            else:
                personality_3class.append('Ambivert')
        
        df['personality_3class'] = personality_3class
        df['extrovert_probability'] = avg_probs
    
    # Analiza
    print("\nDistribution of 3 classes:")
    print(df['personality_3class'].value_counts())
    
    return df

def main(train_path='../../train.csv', test_path='../../test.csv'):
    """Główna funkcja z pełną pipeline GPU
    
    Args:
        train_path: ścieżka do pliku train.csv
        test_path: ścieżka do pliku test.csv
    """
    
    print("🚀 GPU-ACCELERATED PERSONALITY ANALYSIS")
    print("="*60)
    
    # Inicjalizacja
    analyzer = DeepPersonalityAnalyzer()
    
    # 1. Przygotowanie danych
    train_df, test_df = analyzer.prepare_data(train_path, test_path)
    
    # Przygotowanie cech
    feature_cols = [col for col in train_df.columns 
                   if col not in ['id', 'Personality']]
    
    X_train = train_df[feature_cols]
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # 2. Generowanie syntetycznych danych
    X_train_aug, y_train_aug, y_train_3class = analyzer.generate_synthetic_ambiverts(
        X_train, y_train, target_ratio=0.2
    )
    
    # 3. Tworzenie 3 klas na oryginalnych danych
    train_df_with_3class = create_3class_labels(train_df.copy(), X_train, method='neural')
    
    # Mapowanie 3 klas
    class_mapping = {
        'Strong_Introvert': 0,
        'Ambivert': 1,
        'Strong_Extrovert': 2
    }
    y_train_3class_mapped = train_df_with_3class['personality_3class'].map(class_mapping).values
    
    # 4. Optymalizacja hiperparametrów z Optuna
    print("\n🔍 Optimizing hyperparameters with Optuna...")
    
    # XGBoost optimization
    study_xgb = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train_3class_mapped),
        n_trials=50,
        n_jobs=1  # GPU nie wspiera parallel trials
    )
    
    print(f"\nBest XGBoost params: {study_xgb.best_params}")
    print(f"Best XGBoost score: {study_xgb.best_value:.4f}")
    
    # CatBoost optimization
    study_cb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
    study_cb.optimize(
        lambda trial: objective_catboost(trial, X_train, y_train_3class_mapped),
        n_trials=30
    )
    
    print(f"\nBest CatBoost params: {study_cb.best_params}")
    print(f"Best CatBoost score: {study_cb.best_value:.4f}")
    
    # 5. Trenowanie finalnych modeli
    print("\n🚀 Training final ensemble...")
    
    # Podział na train/val
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_3class_mapped,
        test_size=0.2, stratify=y_train_3class_mapped, random_state=42
    )
    
    # XGBoost 3.x z najlepszymi parametrami
    best_xgb_params = study_xgb.best_params.copy()
    best_xgb_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'device': 'cuda:0',  # XGBoost 3.x GPU specification
        'tree_method': 'hist',  # 'hist' automatically uses GPU when device='cuda'
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 10,  # W konstruktorze, nie w fit()!
        'random_state': 42,
        'verbosity': 0
    })
    
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    # CatBoost z najlepszymi parametrami
    best_cb_params = study_cb.best_params.copy()
    best_cb_params.update({
        'task_type': 'GPU',
        'devices': '0:1',
        'loss_function': 'MultiClass',
        'verbose': False,
        'random_state': 42
    })
    
    cb_model = cb.CatBoostClassifier(**best_cb_params)
    cb_model.fit(X_tr, y_tr)
    
    # Neural ensemble
    neural_models = train_neural_ensemble(X_tr, y_tr, X_val, y_val)
    
    # 6. Predykcja na zbiorze testowym
    print("\n📈 Predicting on test set...")
    
    X_test = test_df[feature_cols]
    X_test_scaled = analyzer.scaler.fit_transform(X_test)
    
    # Predykcje z każdego modelu
    xgb_probs = xgb_model.predict_proba(X_test)
    cb_probs = cb_model.predict_proba(X_test)
    
    # Neural predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    neural_probs = []
    for model in neural_models:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            neural_probs.append(probs)
    
    # Ensemble averaging
    all_probs = [xgb_probs, cb_probs] + neural_probs
    ensemble_probs = np.mean(all_probs, axis=0)
    
    # Finalne predykcje
    predictions_3class = np.argmax(ensemble_probs, axis=1)
    
    # Mapowanie z powrotem do nazw klas
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    predictions_3class_names = [reverse_mapping[p] for p in predictions_3class]
    
    # 7. Redukcja do 2 klas
    print("\n🔄 Reducing to 2 classes...")
    
    predictions_2class = []
    confidence_scores = []
    
    for i, pred_3class in enumerate(predictions_3class_names):
        if pred_3class in ['Strong_Introvert', 'Strong_Extrovert']:
            pred_2class = 'Introvert' if 'Introvert' in pred_3class else 'Extrovert'
            confidence = ensemble_probs[i].max()
        else:  # Ambivert
            # Używamy stosunku prawdopodobieństw
            prob_intro = ensemble_probs[i][0]
            prob_extro = ensemble_probs[i][2]
            
            if prob_extro > prob_intro:
                pred_2class = 'Extrovert'
                confidence = prob_extro / (prob_intro + prob_extro)
            else:
                pred_2class = 'Introvert'
                confidence = prob_intro / (prob_intro + prob_extro)
        
        predictions_2class.append(pred_2class)
        confidence_scores.append(confidence)
    
    # 8. Zapis wyników
    results_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction_3class': predictions_3class_names,
        'prediction_2class': predictions_2class,
        'confidence': confidence_scores,
        'prob_strong_intro': ensemble_probs[:, 0],
        'prob_ambivert': ensemble_probs[:, 1],
        'prob_strong_extro': ensemble_probs[:, 2]
    })
    
    results_df.to_csv('gpu_personality_predictions.csv', index=False)
    print("\n✅ Results saved to 'gpu_personality_predictions.csv'")
    
    # Podsumowanie
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Test samples: {len(results_df)}")
    print("\n3-class distribution:")
    print(results_df['prediction_3class'].value_counts())
    print("\n2-class distribution:")
    print(results_df['prediction_2class'].value_counts())
    print(f"\nAverage confidence: {np.mean(confidence_scores):.3f}")
    print(f"Low confidence cases (<0.6): {sum(c < 0.6 for c in confidence_scores)}")
    
    # Zapisz najlepsze parametry
    import json
    best_params = {
        'xgboost': study_xgb.best_params,
        'catboost': study_cb.best_params,
        'scores': {
            'xgboost_cv': study_xgb.best_value,
            'catboost_cv': study_cb.best_value
        }
    }
    
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n✅ Best hyperparameters saved to 'best_hyperparameters.json'")
    
    return results_df

if __name__ == "__main__":
    import sys
    
    # Możliwość podania ścieżek jako argumenty
    if len(sys.argv) >= 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        print(f"Using provided paths: train={train_path}, test={test_path}")
        results = main(train_path, test_path)
    else:
        # Domyślne ścieżki
        print("Using default paths: ../../train.csv, ../../test.csv")
        print("Tip: You can provide custom paths: python script.py path/to/train.csv path/to/test.csv")
        results = main()
