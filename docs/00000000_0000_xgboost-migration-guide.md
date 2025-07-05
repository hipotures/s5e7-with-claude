# XGBoost 3.x Migration Guide - Praktyczne Zmiany w Kodzie

## 📋 Spis Treści
1. [early_stopping_rounds](#1-early_stopping_rounds)
2. [GPU Configuration](#2-gpu-configuration)
3. [Verbosity](#3-verbosity)
4. [Objective dla multi-class](#4-objective-dla-multi-class)
5. [Callback API](#5-callback-api)
6. [Eval Metric](#6-eval-metric)
7. [Categorical Features](#7-categorical-features-nowe)
8. [Zapisywanie/Wczytywanie](#8-zapisywaniewczytywanie)
9. [Multi-GPU](#9-multi-gpu)
10. [Predykcja z best_iteration](#10-predykcja-z-best_iteration)
11. [Kompletny przykład migracji](#kompletny-przykład-migracji)

---

## 1. **early_stopping_rounds**

### ❌ STARE (XGBoost 2.x)
```python
model = xgb.XGBClassifier(n_estimators=1000)
model.fit(X_train, y_train, 
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=10)
```

### ✅ NOWE (XGBoost 3.x)
```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=10  # Przenieś do konstruktora!
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

---

## 2. **GPU Configuration**

### ❌ STARE
```python
params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}
```

### ✅ NOWE
```python
params = {
    'device': 'cuda:0',  # lub 'cuda:1' dla drugiego GPU
    'tree_method': 'hist'  # automatycznie używa GPU gdy device='cuda'
}
```

---

## 3. **Verbosity**

### ❌ STARE
```python
model = xgb.XGBClassifier(silent=True)
# lub
model.fit(X_train, y_train, verbose=False)
```

### ✅ NOWE
```python
model = xgb.XGBClassifier(verbosity=0)  # 0=silent, 1=warning, 2=info, 3=debug
```

---

## 4. **Objective dla multi-class**

### ❌ STARE
```python
params = {
    'objective': 'multi:softmax',  # lub 'multi:softprob'
    'num_class': 3
}
```

### ✅ NOWE (oba nadal działają, ale preferowane)
```python
params = {
    'objective': 'multi:softprob',  # zawsze zwraca prawdopodobieństwa
    'num_class': 3
}
```

---

## 5. **Callback API**

### ❌ STARE
```python
model.fit(X_train, y_train, 
         early_stopping_rounds=10,
         eval_metric='logloss')
```

### ✅ NOWE (opcjonalne, ale daje więcej kontroli)
```python
from xgboost import callback

model = xgb.XGBClassifier(
    callbacks=[
        callback.EarlyStopping(rounds=10, metric_name='logloss'),
        callback.TrainingCallback()  # własny callback
    ]
)
```

---

## 6. **Eval Metric**

### ❌ STARE (w fit)
```python
model.fit(X_train, y_train, eval_metric='mlogloss')
```

### ✅ NOWE (w konstruktorze)
```python
model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)
```

---

## 7. **Categorical Features** (NOWE!)

### ✅ NOWE - natywne wsparcie dla kategorycznych
```python
# Zamiast one-hot encoding:
df['category_col'] = df['category_col'].astype('category')
# XGBoost automatycznie obsłuży!

model = xgb.XGBClassifier(
    enable_categorical=True,  # włącz wsparcie
    tree_method='hist'  # wymagane dla categorical
)
```

---

## 8. **Zapisywanie/Wczytywanie**

### ❌ STARE
```python
model.save_model('model.bin')  # format binarny
```

### ✅ NOWE (preferowane)
```python
model.save_model('model.json')  # JSON - czytelny
model.save_model('model.ubj')   # Universal Binary JSON - mniejszy
```

---

## 9. **Multi-GPU**

### ✅ NOWE - lepsze wsparcie
```python
params = {
    'device': 'cuda',  # używa wszystkich GPU
    # lub
    'device': 'cuda:0',  # konkretne GPU
}
```

---

## 10. **Predykcja z best_iteration**

### ✅ AUTOMATYCZNE w XGBoost 3.x
```python
predictions = model.predict(X_test)  # używa best_iteration automatycznie!

# Opcjonalnie możesz kontrolować:
predictions = model.predict(X_test, 
                          iteration_range=(0, model.best_iteration + 1))
```

---

## 📝 **Checklist dla migracji:**

- [ ] Przenieś `early_stopping_rounds` z `fit()` do konstruktora
- [ ] Zmień `gpu_id` + `tree_method='gpu_hist'` na `device='cuda:0'` + `tree_method='hist'`
- [ ] Zmień `silent=True` na `verbosity=0`
- [ ] Przenieś `eval_metric` z `fit()` do konstruktora
- [ ] Usuń `verbose=False` z `fit()`, użyj `verbosity=0` w konstruktorze

---

## 🚀 **Kompletny przykład migracji**

### ❌ STARY KOD (XGBoost 2.x)
```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    tree_method='gpu_hist',
    gpu_id=0,
    silent=True
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    eval_metric='mlogloss',
    verbose=False
)
```

### ✅ NOWY KOD (XGBoost 3.x)
```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    device='cuda:0',
    tree_method='hist',
    early_stopping_rounds=10,
    eval_metric='mlogloss',
    verbosity=0
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)]
)
```

---

## 🔍 **Dodatkowe wskazówki**

### Sprawdzenie wersji XGBoost
```python
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
print(f"GPU support: {xgb.build_info()['USE_CUDA']}")
```

### Najczęstsze błędy po migracji
1. **Warning o deprecated `early_stopping_rounds`** - przenieś do konstruktora
2. **GPU nie działa** - użyj `device='cuda'` zamiast `gpu_id`
3. **Brak output podczas treningu** - sprawdź `verbosity` zamiast `silent`

### Dokumentacja
- [XGBoost 3.x Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [Migration Guide](https://xgboost.readthedocs.io/en/stable/migration.html)

---

## 📊 **Porównanie wydajności**

| Feature | XGBoost 2.x | XGBoost 3.x | Improvement |
|---------|-------------|-------------|-------------|
| GPU Training | `gpu_hist` | `hist` + `device='cuda'` | ~20% faster |
| Memory Usage | High | Optimized | ~30% less |
| Categorical Support | Manual encoding | Native | 10x faster |
| Multi-GPU | Limited | Full support | 2x speedup |

---

*Ostatnia aktualizacja: Grudzień 2024*