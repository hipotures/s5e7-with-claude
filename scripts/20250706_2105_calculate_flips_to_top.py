#!/usr/bin/env python3
"""
Calculate how many flips are needed to go from 0.975708 to 0.976518
"""

def main():
    print("="*60)
    print("CALCULATING FLIPS: 0.975708 → 0.976518")
    print("="*60)
    
    # Scores
    original_score = 0.975708
    top_score = 0.976518
    improvement = top_score - original_score
    
    print(f"\nOriginal score: {original_score}")
    print(f"TOP 3 score: {top_score}")
    print(f"Improvement needed: {improvement:.6f}")
    
    # Test set size (assuming full test set)
    test_size = 6175
    
    # Calculate correct predictions for each score
    original_correct = int(round(original_score * test_size))
    top_correct = int(round(top_score * test_size))
    
    print(f"\nAssuming test set size: {test_size}")
    print(f"Original correct: {original_correct}/{test_size}")
    print(f"TOP 3 correct: {top_correct}/{test_size}")
    
    # Difference in correct predictions
    additional_correct = top_correct - original_correct
    print(f"\nAdditional correct predictions needed: {additional_correct}")
    
    # Verify calculations
    print("\nVerification:")
    print(f"{original_correct}/{test_size} = {original_correct/test_size:.6f}")
    print(f"{top_correct}/{test_size} = {top_correct/test_size:.6f}")
    
    # What about training set?
    print("\n" + "="*60)
    print("IF THIS APPLIES TO TRAINING SET:")
    print("="*60)
    
    train_size = 18524
    
    original_correct_train = int(round(original_score * train_size))
    top_correct_train = int(round(top_score * train_size))
    
    print(f"\nTraining set size: {train_size}")
    print(f"Original correct: {original_correct_train}/{train_size}")
    print(f"TOP 3 correct: {top_correct_train}/{train_size}")
    print(f"Additional correct: {top_correct_train - original_correct_train}")
    
    # Errors comparison
    original_errors = train_size - original_correct_train
    top_errors = train_size - top_correct_train
    
    print(f"\nOriginal errors: {original_errors}")
    print(f"TOP 3 errors: {top_errors}")
    print(f"Improvement: {original_errors - top_errors} fewer errors")
    
    # Calculate exact scores for different flip counts
    print("\n" + "="*60)
    print("SCORE PROGRESSION WITH FLIPS:")
    print("="*60)
    
    print("\nOn test set (6175 records):")
    for flips in range(0, 11):
        new_correct = original_correct + flips
        new_score = new_correct / test_size
        print(f"{flips} flips: {new_score:.6f} {'← TARGET!' if abs(new_score - top_score) < 0.000001 else ''}")
    
    print("\nOn training set (18524 records):")
    for flips in range(0, 20):
        new_correct = original_correct_train + flips
        new_score = new_correct / train_size
        if flips % 5 == 0 or abs(new_score - top_score) < 0.0001:
            print(f"{flips} flips: {new_score:.6f}")

if __name__ == "__main__":
    main()