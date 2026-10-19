import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

oos = pd.read_csv('reports/india_oos_predictions.csv')
fold_res = pd.read_csv('reports/india_walkforward_fold_results.csv')
comp = pd.read_csv('reports/india_model_comparison.csv')

print('=== INDEPENDENT RECALCULATION OF METRICS ===')
models = ['KNN', 'RF', 'NN']

reconciled_rows = []

for m in models:
    m_col = f'proba_{m.lower()}'
    print(f'\n--- Model: {m} ---')
    
    recalculated_aucs = []
    recalculated_p3 = []
    recalculated_acc = []
    recalculated_prec = []
    recalculated_rec = []
    recalculated_f1 = []
    
    for f in range(1, 9):
        sub = oos[oos['fold'] == f]
        y_true = sub['true_label'].values
        y_prob = sub[m_col].values
        y_pred = (y_prob >= 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_prob)
        recalculated_aucs.append(auc)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        recalculated_acc.append(acc)
        recalculated_prec.append(prec)
        recalculated_rec.append(rec)
        recalculated_f1.append(f1)
        
        p3_list = []
        for d, g in sub.groupby('date'):
            top3 = g.sort_values(m_col, ascending=False).head(3)
            p3_list.append(top3['true_label'].sum() / 3.0)
            
        mean_p3 = np.mean(p3_list)
        recalculated_p3.append(mean_p3)
        
        reported_fold = fold_res[(fold_res['fold'] == f) & (fold_res['model'] == m)].iloc[0]
        rep_auc = reported_fold['roc_auc']
        rep_p3 = reported_fold['precision_at_3']
        auc_match = np.isclose(auc, rep_auc, atol=1e-4)
        p3_match = np.isclose(mean_p3, rep_p3, atol=1e-4)
        print(f"Fold {f}: Recalc AUC={auc:.4f} vs Rep {rep_auc:.4f} (Match: {auc_match}) | Recalc Prec@3={mean_p3:.4f} vs Rep {rep_p3:.4f} (Match: {p3_match})")
        
    mean_auc = np.mean(recalculated_aucs)
    med_auc = np.median(recalculated_aucs)
    std_auc = np.std(recalculated_aucs, ddof=1)
    min_auc = np.min(recalculated_aucs)
    max_auc = np.max(recalculated_aucs)
    pooled_auc = roc_auc_score(oos['true_label'], oos[m_col])
    
    rep_comp = comp[comp['model'] == m].iloc[0]
    print(f"Aggregate: Recalc MeanAUC={mean_auc:.4f} vs Rep {rep_comp['mean_auc']:.4f}")
    print(f"Aggregate: Recalc MedAUC={med_auc:.4f} vs Rep {rep_comp['median_auc']:.4f}")
    print(f"Aggregate: Recalc PooledAUC={pooled_auc:.4f} vs Rep {rep_comp['pooled_oos_auc']:.4f}")

    # Backtest recalculation
    oos_dates = sorted(oos['date'].unique())
    rebal_dates = oos_dates[::10]
    
    fee_per_trade = 0.0015
    port_rets_gross = []
    port_rets_net = []
    bench_rets = []
    prev_stocks = set()
    
    for d in rebal_dates:
        day_preds = oos[oos['date'] == d].sort_values(m_col, ascending=False)
        top3 = day_preds.head(3)
        current_stocks = set(top3['stock'].tolist())
        
        if not prev_stocks:
            turnover = 1.0
        else:
            turnover = len(current_stocks - prev_stocks) / 3.0
        prev_stocks = current_stocks
        
        gross_ret = top3['fwd_ret'].mean()
        costs = (turnover * 2.0) * fee_per_trade
        net_ret = gross_ret - costs
        b_ret = day_preds['fwd_ret'].mean()
        
        port_rets_gross.append(gross_ret)
        port_rets_net.append(net_ret)
        bench_rets.append(b_ret)
        
    port_rets_gross = np.array(port_rets_gross)
    port_rets_net = np.array(port_rets_net)
    bench_rets = np.array(bench_rets)
    
    cum_gross = (np.prod(1.0 + port_rets_gross) - 1.0) * 100.0
    cum_net = (np.prod(1.0 + port_rets_net) - 1.0) * 100.0
    cum_bench = (np.prod(1.0 + bench_rets) - 1.0) * 100.0
    
    n_periods = len(port_rets_net)
    periods_per_year = 25.2
    years = n_periods / periods_per_year
    ann_net = ((1.0 + cum_net / 100.0) ** (1.0 / years) - 1.0) * 100.0
    vol_net = np.std(port_rets_net, ddof=1) * np.sqrt(periods_per_year) * 100.0
    sharpe_net = (ann_net / vol_net) if vol_net > 0 else 0.0
    
    wealth = np.cumprod(1.0 + port_rets_net)
    peaks = np.maximum.accumulate(wealth)
    dd = (wealth - peaks) / peaks
    mdd = np.min(dd) * 100.0
    
    print(f"Backtest: Recalc CumGross={cum_gross:.2f}% vs Rep {rep_comp['strat_cum_return_gross']:.2f}%")
    print(f"Backtest: Recalc CumNet={cum_net:.2f}% vs Rep {rep_comp['strat_cum_return_net']:.2f}%")
    print(f"Backtest: Recalc AnnNet={ann_net:.2f}% vs Rep {rep_comp['strat_ann_return_net']:.2f}%")
    print(f"Backtest: Recalc VolNet={vol_net:.2f}% vs Rep {rep_comp['strat_ann_volatility']:.2f}%")
    print(f"Backtest: Recalc Sharpe={sharpe_net:.2f} vs Rep {rep_comp['strat_sharpe_ratio']:.2f}")
    print(f"Backtest: Recalc MDD={mdd:.2f}% vs Rep {rep_comp['strat_max_drawdown']:.2f}%")
    print(f"Backtest: Recalc CumBench={cum_bench:.2f}% vs Rep {rep_comp['bench_cum_return']:.2f}%")
    
    reconciled_rows.append({
        'model': m,
        'recalc_mean_auc': round(mean_auc, 4),
        'reported_mean_auc': rep_comp['mean_auc'],
        'recalc_median_auc': round(med_auc, 4),
        'reported_median_auc': rep_comp['median_auc'],
        'recalc_pooled_auc': round(pooled_auc, 4),
        'reported_pooled_auc': rep_comp['pooled_oos_auc'],
        'recalc_mean_prec_at_3': round(np.mean(recalculated_p3), 4),
        'reported_mean_prec_at_3': rep_comp['mean_precision_at_3'],
        'recalc_cum_gross_pct': round(cum_gross, 2),
        'reported_cum_gross_pct': rep_comp['strat_cum_return_gross'],
        'recalc_cum_net_pct': round(cum_net, 2),
        'reported_cum_net_pct': rep_comp['strat_cum_return_net'],
        'recalc_ann_net_pct': round(ann_net, 2),
        'reported_ann_net_pct': rep_comp['strat_ann_return_net'],
        'recalc_vol_net_pct': round(vol_net, 2),
        'reported_vol_net_pct': rep_comp['strat_ann_volatility'],
        'recalc_sharpe': round(sharpe_net, 2),
        'reported_sharpe': rep_comp['strat_sharpe_ratio'],
        'recalc_mdd_pct': round(mdd, 2),
        'reported_mdd_pct': rep_comp['strat_max_drawdown'],
        'recalc_cum_bench_pct': round(cum_bench, 2),
        'reported_cum_bench_pct': rep_comp['bench_cum_return']
    })

pd.DataFrame(reconciled_rows).to_csv('reports/phase2_reconciled_metrics.csv', index=False)
print('\nSaved reports/phase2_reconciled_metrics.csv successfully.')
