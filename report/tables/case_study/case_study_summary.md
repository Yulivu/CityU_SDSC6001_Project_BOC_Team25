## Case Study (MA-GNN variance-based aspect weighting)

Selected user:
- user_id: 4kAKOaQL3W5z5nFXhhdvyQ
- source city → target city: New Orleans → Metairie
- source-city reviews: 32

Aspect weights (w_k) and source-city rating variance:
- food: w_k=0.8337, variance=0.638 (n=28)
- service: w_k=0.1663, variance=2.25 (n=4)
- atmosphere: w_k=0.0000 (n=0)
- price: w_k=0.0000 (n=0)

Key observations:
- The variance-based weighting assigns a dominant weight to food, indicating stable food preference in the source city (low variance) compared to service (high variance).
- Ground-truth retrieval improves under MA-GNN in the target city: best ground-truth rank is 1 for MA-GNN vs 2 for the Popularity baseline (Top-10).

Figures and tables:
- Figure (w_k + variance): report/figures/case_study/wk_variance.png
- Figure (Top-10 scores): report/figures/case_study/magnn_top10_scores.png
- Figure (GT rank comparison): report/figures/case_study/gt_rank_compare.png
- Table (MA-GNN Top-10): report/tables/case_study/top10_magnn.csv
- Table (Popularity Top-10): report/tables/case_study/top10_popularity.csv

