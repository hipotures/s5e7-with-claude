#!/usr/bin/env python3
"""
Calculate how many flips needed for 100% accuracy
"""

def main():
    print("="*60)
    print("CALCULATING FLIPS TO REACH 100% ACCURACY")
    print("="*60)
    
    # Current best scores
    your_score = 0.975708
    top3_score = 0.976518
    perfect_score = 1.0
    
    # Test set calculations
    test_size = 6175
    
    print("\nON TEST SET (6175 records):")
    print("-"*40)
    
    # Your current state
    your_correct = int(round(your_score * test_size))
    your_errors = test_size - your_correct
    
    print(f"Your score: {your_score:.6f}")
    print(f"Correct: {your_correct}/{test_size}")
    print(f"Errors: {your_errors}")
    print(f"\nFlips needed for 100%: {your_errors}")
    
    # TOP 3 state
    top3_correct = int(round(top3_score * test_size))
    top3_errors = test_size - top3_correct
    
    print(f"\nTOP 3 score: {top3_score:.6f}")
    print(f"Correct: {top3_correct}/{test_size}")
    print(f"Errors: {top3_errors}")
    print(f"Flips needed for 100%: {top3_errors}")
    
    # Training set calculations
    print("\n" + "="*60)
    print("ON TRAINING SET (18524 records):")
    print("-"*40)
    
    train_size = 18524
    
    # Your state on training
    your_correct_train = int(round(your_score * train_size))
    your_errors_train = train_size - your_correct_train
    
    print(f"Your score: {your_score:.6f}")
    print(f"Correct: {your_correct_train}/{train_size}")
    print(f"Errors: {your_errors_train}")
    print(f"\nFlips needed for 100%: {your_errors_train}")
    
    # TOP 3 on training
    top3_correct_train = int(round(top3_score * train_size))
    top3_errors_train = train_size - top3_correct_train
    
    print(f"\nTOP 3 score: {top3_score:.6f}")
    print(f"Correct: {top3_correct_train}/{train_size}")
    print(f"Errors: {top3_errors_train}")
    print(f"Flips needed for 100%: {top3_errors_train}")
    
    # Theoretical limits
    print("\n" + "="*60)
    print("THEORETICAL ANALYSIS:")
    print("="*60)
    
    print("\nIf 2.43% are true ambiverts (can't be classified):")
    ambiverts_test = int(test_size * 0.0243)
    ambiverts_train = int(train_size * 0.0243)
    
    print(f"Test set: {ambiverts_test} ambiverts")
    print(f"Max possible accuracy: {(test_size - ambiverts_test)/test_size:.6f}")
    
    print(f"\nTraining set: {ambiverts_train} ambiverts")
    print(f"Max possible accuracy: {(train_size - ambiverts_train)/train_size:.6f}")
    
    # Show progression
    print("\n" + "="*60)
    print("SCORE PROGRESSION (TEST SET):")
    print("="*60)
    
    milestones = [0.98, 0.985, 0.99, 0.995, 0.999, 1.0]
    
    for target in milestones:
        needed_correct = int(target * test_size)
        current_correct = your_correct
        flips_needed = needed_correct - current_correct
        print(f"{target:.1%}: need {flips_needed} flips ({needed_correct}/{test_size})")
    
    # Reality check
    print("\n" + "="*60)
    print("REALITY CHECK:")
    print("="*60)
    
    print(f"\nYou need {your_errors} perfect flips for 100%")
    print(f"TOP 3 need {top3_errors} perfect flips for 100%")
    print(f"\nTOP 3 are only {top3_errors}/{your_errors} = {top3_errors/your_errors:.1%} of the way from you to perfection!")

if __name__ == "__main__":
    main()