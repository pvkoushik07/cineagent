"""
Create visualizations for final report.

Generates charts from evaluation results:
- Variant comparison (A vs B vs C)
- Performance by query family
- Latency comparison
- Ablation study results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Load results
results_file = Path("data/results/eval_results.json")
with open(results_file) as f:
    results = json.load(f)

# Create output directory
output_dir = Path("data/results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Overall Variant Comparison ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

variants = ['Variant A\n(Plain LLM)', 'Variant B\n(Fixed RAG)', 'Variant C\n(Full Agent)']
recalls = [
    results['variant_A']['summary']['overall']['recall_at_5'],
    results['variant_B']['summary']['overall']['recall_at_5'],
    results['variant_C']['summary']['overall']['recall_at_5'],
]

colors = ['#E74C3C', '#3498DB', '#2ECC71']
bars = ax.bar(variants, recalls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, recall in zip(bars, recalls):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{recall*100:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Recall@5', fontweight='bold', fontsize=12)
ax.set_title('System Variant Comparison: Overall Performance', fontweight='bold', fontsize=14)
ax.set_ylim(0, 0.7)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "fig1_variant_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fig1_variant_comparison.png'}")
plt.close()

# ── Figure 2: Performance by Query Family ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

families = ['Factual', 'Visual', 'Multi-hop']
variant_a_recalls = [
    results['variant_A']['summary']['factual']['recall_at_5'],
    results['variant_A']['summary']['visual']['recall_at_5'],
    results['variant_A']['summary']['multi_hop']['recall_at_5'],
]
variant_b_recalls = [
    results['variant_B']['summary']['factual']['recall_at_5'],
    results['variant_B']['summary']['visual']['recall_at_5'],
    results['variant_B']['summary']['multi_hop']['recall_at_5'],
]
variant_c_recalls = [
    results['variant_C']['summary']['factual']['recall_at_5'],
    results['variant_C']['summary']['visual']['recall_at_5'],
    results['variant_C']['summary']['multi_hop']['recall_at_5'],
]

x = np.arange(len(families))
width = 0.25

bars1 = ax.bar(x - width, variant_a_recalls, width, label='Variant A (Plain LLM)',
               color='#E74C3C', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, variant_b_recalls, width, label='Variant B (Fixed RAG)',
               color='#3498DB', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, variant_c_recalls, width, label='Variant C (Full Agent)',
               color='#2ECC71', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                    f'{height*100:.0f}%',
                    ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Recall@5', fontweight='bold', fontsize=12)
ax.set_title('Performance by Query Family', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(families)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "fig2_query_family_performance.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fig2_query_family_performance.png'}")
plt.close()

# ── Figure 3: Latency Comparison ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

variants = ['Variant A\n(Plain LLM)', 'Variant B\n(Fixed RAG)', 'Variant C\n(Full Agent)']
latencies = [
    results['variant_A']['summary']['overall']['mean_latency_ms'] / 1000,  # Convert to seconds
    results['variant_B']['summary']['overall']['mean_latency_ms'] / 1000,
    results['variant_C']['summary']['overall']['mean_latency_ms'] / 1000,
]

colors = ['#E74C3C', '#3498DB', '#2ECC71']
bars = ax.bar(variants, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, latency in zip(bars, latencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{latency:.1f}s',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Mean Latency (seconds)', fontweight='bold', fontsize=12)
ax.set_title('Query Latency Comparison', fontweight='bold', fontsize=14)
ax.set_ylim(0, max(latencies) * 1.2)

plt.tight_layout()
plt.savefig(output_dir / "fig3_latency_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fig3_latency_comparison.png'}")
plt.close()

# ── Figure 4: Ablation Study - Retrieval Strategies ───────────────────────────
# Parse ablation 1 results
ablation_1_results = {}
for query_result in results['ablation_1']['per_query']:
    variant = query_result['variant'].replace('ablation1_', '')
    family = query_result['query_family']
    recall = query_result['recall_at_5']

    if variant not in ablation_1_results:
        ablation_1_results[variant] = {}
    if family not in ablation_1_results[variant]:
        ablation_1_results[variant][family] = []
    ablation_1_results[variant][family].append(recall)

# Compute mean recall per variant per family
ablation_means = {}
for variant in ['text_only', 'caption_only', 'clip_only', 'hybrid_rrf']:
    ablation_means[variant] = {}
    for family in ['factual', 'visual']:
        if family in ablation_1_results[variant]:
            ablation_means[variant][family] = np.mean(ablation_1_results[variant][family])
        else:
            ablation_means[variant][family] = 0.0

fig, ax = plt.subplots(figsize=(12, 6))

strategies = ['Text Only', 'Caption Only', 'CLIP Only', 'Hybrid RRF']
strategy_keys = ['text_only', 'caption_only', 'clip_only', 'hybrid_rrf']

factual_recalls = [ablation_means[k]['factual'] for k in strategy_keys]
visual_recalls = [ablation_means[k]['visual'] for k in strategy_keys]

x = np.arange(len(strategies))
width = 0.35

bars1 = ax.bar(x - width/2, factual_recalls, width, label='Factual Queries',
               color='#9B59B6', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, visual_recalls, width, label='Visual Queries',
               color='#E67E22', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                    f'{height*100:.0f}%',
                    ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Recall@5', fontweight='bold', fontsize=12)
ax.set_title('Ablation Study: Retrieval Strategy Performance', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(output_dir / "fig4_ablation_retrieval.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fig4_ablation_retrieval.png'}")
plt.close()

# ── Figure 5: Key Finding Highlight ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# Create comparison between B and C for multi-hop
comparison_data = {
    'Variant': ['Fixed RAG\n(Variant B)', 'Full Agent\n(Variant C)'],
    'Multi-hop Recall': [
        results['variant_B']['summary']['multi_hop']['recall_at_5'] * 100,
        results['variant_C']['summary']['multi_hop']['recall_at_5'] * 100,
    ],
    'Overall Recall': [
        results['variant_B']['summary']['overall']['recall_at_5'] * 100,
        results['variant_C']['summary']['overall']['recall_at_5'] * 100,
    ]
}

x = np.arange(len(comparison_data['Variant']))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_data['Multi-hop Recall'], width,
               label='Multi-hop Queries', color='#E74C3C', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, comparison_data['Overall Recall'], width,
               label='Overall Performance', color='#3498DB', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Recall@5 (%)', fontweight='bold', fontsize=12)
ax.set_title('Key Finding: Fixed RAG Outperforms Full Agent', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(comparison_data['Variant'])
ax.legend(fontsize=10)
ax.set_ylim(0, 80)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='50% threshold')

# Add annotation
ax.annotate('Agentic routing decreases\nmulti-hop performance',
            xy=(1, comparison_data['Multi-hop Recall'][1]),
            xytext=(0.3, comparison_data['Multi-hop Recall'][0] - 10),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / "fig5_key_finding.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fig5_key_finding.png'}")
plt.close()

print("\n✅ All visualizations created successfully!")
print(f"📁 Output directory: {output_dir.absolute()}")
