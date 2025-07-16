#!/usr/bin/env python3
"""
Extract cluster-based metamodel results from log file.
"""

import re

# Read log file
with open('output/cluster_flip_evaluation.log', 'r') as f:
    content = f.read()

# Parse results
pattern = r'Evaluating (\d+)/20: Index (\d+).*?Mean improvement: ([+-]\d+\.\d+) ± (\d+\.\d+).*?Positive in (\d+)/9 clusters'
matches = re.findall(pattern, content, re.DOTALL)

# Process results
results = []
for match in matches:
    eval_num, idx, mean_imp, std_imp, pos_clusters = match
    results.append({
        'eval_num': int(eval_num),
        'idx': int(idx),
        'mean_improvement': float(mean_imp),
        'std_improvement': float(std_imp),
        'positive_clusters': int(pos_clusters)
    })

# Show positive improvements
positive_results = [r for r in results if r['mean_improvement'] > 0]
positive_results.sort(key=lambda x: x['mean_improvement'], reverse=True)

print("="*60)
print("CLUSTER-BASED METAMODEL RESULTS")
print("="*60)
print(f"\nTotal evaluations: {len(results)}")
print(f"Positive improvements: {len(positive_results)}")

if positive_results:
    print("\nTop positive improvements:")
    print("-"*60)
    print(f"{'Rank':<5} {'Index':<10} {'Mean Imp.':<12} {'Std Dev.':<12} {'Pos. Clusters':<12}")
    print("-"*60)
    
    for i, result in enumerate(positive_results[:10]):
        print(f"{i+1:<5} {result['idx']:<10} "
              f"{result['mean_improvement']:+.6f}    "
              f"{result['std_improvement']:.6f}    "
              f"{result['positive_clusters']}/9")
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(positive_results)
    df.to_csv('output/cluster_positive_improvements.csv', index=False)
    print(f"\nSaved {len(positive_results)} positive results to output/cluster_positive_improvements.csv")

print("\nSummary:")
print(f"- {len(positive_results)}/{len(results)} evaluations showed positive improvement")
print(f"- Best improvement: {positive_results[0]['mean_improvement']:+.6f} (Index {positive_results[0]['idx']})")
print(f"- Average positive improvement: {sum(r['mean_improvement'] for r in positive_results)/len(positive_results):.6f}")

# Show which clusters tend to benefit
all_positive_clusters = []
for result in positive_results:
    if result['positive_clusters'] > 0:
        all_positive_clusters.extend([result['idx']] * result['positive_clusters'])

print(f"\nTotal positive cluster evaluations: {len(all_positive_clusters)}")

print("\n" + "="*60)