from train import grid_search

results = grid_search.cv_results_

print("=== Grid Search Results ===\n")

for params, f1, f05, f2, rank in zip(
    results['params'],
    results['mean_test_f1'],
    results['mean_test_f0_5'],
    results['mean_test_f2'],
    results['rank_test_f1']
):
    print(
        f"alpha={params['alpha']}, "
        f"fit_prior={params['fit_prior']}, "
        f"F1={f1:.4f}, "
        f"F0.5={f05:.4f}, "
        f"F2={f2:.4f}, "
        f"Rank={rank}"
    )