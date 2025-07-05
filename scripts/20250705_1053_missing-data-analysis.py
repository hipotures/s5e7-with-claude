import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def analyze_missing_bias(train_df, test_df, predictions_df):
    """
    Głęboka analiza biasu związanego z brakami danych
    """
    print("🔍 ANALIZA BIASU DLA REKORDÓW BEZ BRAKÓW")
    print("="*60)
    
    # 1. Sprawdź rzeczywisty rozkład w danych treningowych
    print("\n1️⃣ DANE TRENINGOWE - Rzeczywisty rozkład:")
    
    # Dodaj kolumnę z liczbą braków
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                   'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                   'Post_frequency']
    
    train_df['total_missing'] = train_df[feature_cols].isnull().sum(axis=1)
    
    # Analiza dla różnych poziomów braków
    for missing_count in range(5):
        mask = train_df['total_missing'] == missing_count
        if mask.sum() > 0:
            intro_ratio = (train_df.loc[mask, 'Personality'] == 'Introvert').mean()
            print(f"  {missing_count} braków: {mask.sum():5d} rekordów, {intro_ratio:.1%} introwertyków")
    
    # 2. Porównaj z predykcjami
    print("\n2️⃣ PREDYKCJE - Co przewiduje model:")
    
    # Dodaj braki do test_df
    test_df['total_missing'] = test_df[feature_cols].isnull().sum(axis=1)
    
    # Merge z predykcjami
    test_with_pred = test_df.merge(predictions_df[['id', 'prediction_2class']], on='id')
    
    for missing_count in range(5):
        mask = test_with_pred['total_missing'] == missing_count
        if mask.sum() > 0:
            intro_pred_ratio = (test_with_pred.loc[mask, 'prediction_2class'] == 'Introvert').mean()
            print(f"  {missing_count} braków: {mask.sum():5d} rekordów, {intro_pred_ratio:.1%} przewidzianych jako introwertycy")
    
    # 3. Analiza per cecha
    print("\n3️⃣ WPŁYW POJEDYNCZYCH BRAKÓW:")
    
    results = []
    for col in feature_cols:
        # Rekordy z brakiem w tej kolumnie
        mask_missing = test_with_pred[col].isnull()
        mask_present = ~mask_missing
        
        if mask_missing.sum() > 0 and mask_present.sum() > 0:
            intro_ratio_missing = (test_with_pred.loc[mask_missing, 'prediction_2class'] == 'Introvert').mean()
            intro_ratio_present = (test_with_pred.loc[mask_present, 'prediction_2class'] == 'Introvert').mean()
            
            lift = intro_ratio_missing / (intro_ratio_present + 0.001)
            
            results.append({
                'feature': col,
                'n_missing': mask_missing.sum(),
                'intro_ratio_missing': intro_ratio_missing,
                'intro_ratio_present': intro_ratio_present,
                'lift': lift
            })
    
    results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
    print(results_df.to_string(index=False))
    
    # 4. Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 4.1 Rozkład braków w train vs predykcje
    ax = axes[0, 0]
    missing_train = train_df.groupby('total_missing')['Personality'].apply(
        lambda x: (x == 'Introvert').mean()
    )
    missing_pred = test_with_pred.groupby('total_missing')['prediction_2class'].apply(
        lambda x: (x == 'Introvert').mean()
    )
    
    x = range(len(missing_train))
    width = 0.35
    ax.bar([i - width/2 for i in x], missing_train.values, width, label='Train (rzeczywiste)', alpha=0.8)
    ax.bar([i + width/2 for i in x[:len(missing_pred)]], missing_pred.values, width, label='Test (predykcje)', alpha=0.8)
    ax.set_xlabel('Liczba braków')
    ax.set_ylabel('% Introwertyków')
    ax.set_title('Rozkład introwertyków vs liczba braków')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(missing_train.index)
    
    # 4.2 Lift per cecha
    ax = axes[0, 1]
    results_df_sorted = results_df.sort_values('lift')
    ax.barh(results_df_sorted['feature'], results_df_sorted['lift'], color='green')
    ax.axvline(x=1, color='red', linestyle='--', label='Brak wpływu')
    ax.set_xlabel('Lift (ile razy więcej introwertyków)')
    ax.set_title('Wpływ braku w danej cesze na predykcję introwertyzmu')
    
    # 4.3 Confusion matrix dla rekordów bez braków
    ax = axes[1, 0]
    no_missing_mask = test_with_pred['total_missing'] == 0
    if no_missing_mask.sum() > 100:  # Jeśli mamy dość danych
        # Zakładamy że mamy ground truth w jakiejś formie
        # Jeśli nie, to pokażemy tylko rozkład predykcji
        pred_counts = test_with_pred.loc[no_missing_mask, 'prediction_2class'].value_counts()
        ax.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        ax.set_title(f'Predykcje dla rekordów bez braków (n={no_missing_mask.sum()})')
    
    # 4.4 Analiza prawdopodobieństw
    ax = axes[1, 1]
    if 'extrovert_prob_2class' in test_with_pred.columns:
        for missing_count in range(3):
            mask = test_with_pred['total_missing'] == missing_count
            if mask.sum() > 50:
                ax.hist(test_with_pred.loc[mask, 'extrovert_prob_2class'], 
                       bins=30, alpha=0.5, label=f'{missing_count} braków', density=True)
        ax.set_xlabel('P(Extrovert)')
        ax.set_ylabel('Gęstość')
        ax.set_title('Rozkład prawdopodobieństw vs liczba braków')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('missing_data_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_with_pred, results_df

def create_postprocessing_rules(test_with_pred, results_df, predictions_df):
    """
    Tworzy reguły post-processingu do poprawy predykcji
    """
    print("\n\n🔧 POST-PROCESSING RULES")
    print("="*60)
    
    # Kopiuj oryginalne predykcje
    predictions_improved = predictions_df.copy()
    predictions_improved['original_prediction'] = predictions_improved['prediction_2class']
    predictions_improved['postprocessed'] = False
    
    # Merge z danymi testowymi
    test_data = test_with_pred[['id', 'total_missing'] + 
                               [col for col in test_with_pred.columns if col.endswith('_missing')]].copy()
    
    predictions_improved = predictions_improved.merge(test_data, on='id', how='left')
    
    # REGUŁA 1: Drained_after_socializing
    # Jeśli brak i prawdopodobieństwo 0.3-0.5 → Introvert
    print("\n📌 Reguła 1: Brak w Drained_after_socializing")
    if 'Drained_after_socializing' in test_with_pred.columns:
        mask = (
            (test_with_pred['Drained_after_socializing'].isnull()) &
            (predictions_improved['extrovert_prob_2class'] > 0.3) &
            (predictions_improved['extrovert_prob_2class'] < 0.5)
        )
        n_changed = mask.sum()
        predictions_improved.loc[mask, 'prediction_2class'] = 'Introvert'
        predictions_improved.loc[mask, 'postprocessed'] = True
        print(f"   Zmieniono {n_changed} rekordów na Introvert")
    
    # REGUŁA 2: 2+ braków z granicznym prawdopodobieństwem
    print("\n📌 Reguła 2: 2+ braków z granicznym prawdopodobieństwem")
    mask = (
        (predictions_improved['total_missing'] >= 2) &
        (predictions_improved['extrovert_prob_2class'] > 0.45) &
        (predictions_improved['extrovert_prob_2class'] < 0.55) &
        (~predictions_improved['postprocessed'])  # Nie zmieniaj już zmienionych
    )
    n_changed = mask.sum()
    predictions_improved.loc[mask, 'prediction_2class'] = 'Introvert'
    predictions_improved.loc[mask, 'postprocessed'] = True
    print(f"   Zmieniono {n_changed} rekordów na Introvert")
    
    # REGUŁA 3: Stage_fear + wysokie anxiety
    print("\n📌 Reguła 3: Brak w Stage_fear + wysokie anxiety indicators")
    if 'Stage_fear' in test_with_pred.columns and 'anxiety_score' in predictions_improved.columns:
        mask = (
            (test_with_pred['Stage_fear'].isnull()) &
            (predictions_improved.get('anxiety_score', 0) > predictions_improved.get('anxiety_score', 0).median()) &
            (predictions_improved['extrovert_prob_2class'] > 0.4) &
            (predictions_improved['extrovert_prob_2class'] < 0.6) &
            (~predictions_improved['postprocessed'])
        )
        n_changed = mask.sum()
        predictions_improved.loc[mask, 'prediction_2class'] = 'Introvert'
        predictions_improved.loc[mask, 'postprocessed'] = True
        print(f"   Zmieniono {n_changed} rekordów na Introvert")
    
    # REGUŁA 4: Korekta dla 0 braków
    print("\n📌 Reguła 4: Balansowanie rekordów bez braków")
    # Dla rekordów bez braków, obniż próg dla introwertyzmu
    mask = (
        (predictions_improved['total_missing'] == 0) &
        (predictions_improved['extrovert_prob_2class'] < 0.65) &  # Bardziej liberalny próg
        (predictions_improved['extrovert_prob_2class'] > 0.35) &
        (~predictions_improved['postprocessed'])
    )
    # Tylko część z nich zmień (żeby nie przesadzić)
    if mask.sum() > 0:
        # Zmień te z najniższym prawdopodobieństwem ekstrawertyzmu
        candidates = predictions_improved.loc[mask].nsmallest(
            int(mask.sum() * 0.3), 'extrovert_prob_2class'
        )
        predictions_improved.loc[candidates.index, 'prediction_2class'] = 'Introvert'
        predictions_improved.loc[candidates.index, 'postprocessed'] = True
        print(f"   Zmieniono {len(candidates)} rekordów na Introvert")
    
    # Podsumowanie
    print("\n📊 PODSUMOWANIE POST-PROCESSINGU:")
    print(f"Całkowita liczba zmienionych predykcji: {predictions_improved['postprocessed'].sum()}")
    
    # Porównanie rozkładów
    orig_intro = (predictions_improved['original_prediction'] == 'Introvert').mean()
    new_intro = (predictions_improved['prediction_2class'] == 'Introvert').mean()
    print(f"% Introwertyków przed: {orig_intro:.1%}")
    print(f"% Introwertyków po:    {new_intro:.1%}")
    
    # Analiza zmian per liczba braków
    print("\nZmiany per liczba braków:")
    for n_missing in range(5):
        mask = predictions_improved['total_missing'] == n_missing
        if mask.sum() > 0:
            changed = (
                predictions_improved.loc[mask, 'original_prediction'] != 
                predictions_improved.loc[mask, 'prediction_2class']
            ).sum()
            print(f"  {n_missing} braków: {changed}/{mask.sum()} zmian")
    
    return predictions_improved

def validate_postprocessing(predictions_improved):
    """
    Walidacja i wizualizacja efektów post-processingu
    """
    print("\n\n✅ WALIDACJA POST-PROCESSINGU")
    print("="*60)
    
    # Statystyki zmian
    changes = predictions_improved['postprocessed'].sum()
    total = len(predictions_improved)
    print(f"Zmieniono {changes}/{total} predykcji ({changes/total:.1%})")
    
    # Które rekordy zostały zmienione?
    changed_mask = predictions_improved['postprocessed']
    
    print("\nCharakterystyka zmienionych rekordów:")
    print(f"Średnia liczba braków: {predictions_improved.loc[changed_mask, 'total_missing'].mean():.2f}")
    print(f"Średnie prawdopodobieństwo (przed zmianą): {predictions_improved.loc[changed_mask, 'extrovert_prob_2class'].mean():.3f}")
    
    # Wizualizacja
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Rozkład prawdopodobieństw dla zmienionych rekordów
    ax = axes[0]
    ax.hist(predictions_improved.loc[changed_mask, 'extrovert_prob_2class'], 
           bins=30, alpha=0.7, color='orange')
    ax.axvline(x=0.5, color='red', linestyle='--')
    ax.set_xlabel('P(Extrovert) przed zmianą')
    ax.set_ylabel('Liczba rekordów')
    ax.set_title(f'Rozkład prawdopodobieństw dla {changes} zmienionych rekordów')
    
    # 2. Zmiany per liczba braków
    ax = axes[1]
    changes_by_missing = predictions_improved.groupby('total_missing')['postprocessed'].agg(['sum', 'count'])
    changes_by_missing['ratio'] = changes_by_missing['sum'] / changes_by_missing['count']
    ax.bar(changes_by_missing.index, changes_by_missing['ratio'])
    ax.set_xlabel('Liczba braków')
    ax.set_ylabel('% zmienionych predykcji')
    ax.set_title('Procent zmian per liczba braków')
    
    # 3. Przed vs Po - rozkład introwertyków
    ax = axes[2]
    before = predictions_improved.groupby('total_missing')['original_prediction'].apply(
        lambda x: (x == 'Introvert').mean()
    )
    after = predictions_improved.groupby('total_missing')['prediction_2class'].apply(
        lambda x: (x == 'Introvert').mean()
    )
    
    x = range(len(before))
    width = 0.35
    ax.bar([i - width/2 for i in x], before.values, width, label='Przed', alpha=0.8)
    ax.bar([i + width/2 for i in x], after.values, width, label='Po', alpha=0.8)
    ax.set_xlabel('Liczba braków')
    ax.set_ylabel('% Introwertyków')
    ax.set_title('Wpływ post-processingu na rozkład')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(before.index)
    
    plt.tight_layout()
    plt.savefig('postprocessing_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Zapisz ulepszone predykcje
    output_df = predictions_improved[['id', 'prediction_3class', 'prediction_2class', 
                                     'confidence', 'postprocessed']]
    output_df.to_csv('predictions_postprocessed.csv', index=False)
    print("\n💾 Zapisano ulepszone predykcje do 'predictions_postprocessed.csv'")
    
    return predictions_improved

# Główna funkcja
def improve_predictions_with_missing_pattern(train_path, test_path, predictions_path):
    """
    Główna funkcja do analizy i poprawy predykcji
    """
    print("🚀 ANALIZA I POPRAWA PREDYKCJI NA PODSTAWIE WZORCÓW BRAKÓW")
    print("="*70)
    
    # Wczytaj dane
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    predictions_df = pd.read_csv(predictions_path)
    
    # 1. Analiza biasu
    test_with_pred, results_df = analyze_missing_bias(train_df, test_df, predictions_df)
    
    # 2. Post-processing
    predictions_improved = create_postprocessing_rules(test_with_pred, results_df, predictions_df)
    
    # 3. Walidacja
    validate_postprocessing(predictions_improved)
    
    return predictions_improved

# Przykład użycia
if __name__ == "__main__":
    # Ścieżki do plików
    train_path = "../../train.csv"
    test_path = "../../test.csv"
    predictions_path = "gpu_personality_predictions.csv"  # Twoje predykcje
    
    # Uruchom analizę i poprawę
    improved_predictions = improve_predictions_with_missing_pattern(
        train_path, test_path, predictions_path
    )