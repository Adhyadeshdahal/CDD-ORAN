# Step 1 v1 result (frozen protocol f9d87740): KILL

Official run: local detached re-run of the frozen analyzer (the Colab job was lost), 2026-09-30 00:45 NST,
1805 s. JSON: step1_v1_full.json; GT tables: step1_v1_gt_tables.json. Diagnosis: .tmp/PLAN.md + MORNING_BRIEF_3.md.

- K0 PASS (placebo rate .050, 0 BY); K1 PASS (228 sleep units, 62 rejects); G PASS (sleep->nbr pv TRUE +15.4).
- GT (dir): 27 TRUE / 19 NULL / 14 INDET; 8 nbr TRUE; sleep->far all TRUE.
- P1 FAIL (indirect recall 3/8 = .375 < 2/3; precision 1.00, F1 .60, sign .92, far 1).
- P2 FAIL (pooled MSCR indirect F1 .55 vs Granger .75; 0/3 folds).

```
        method   split  ind P  ind R ind F1     F1   sign  far  #dec
      mscr_crt  pooled   1.00   0.38   0.55   0.60   0.92    1    13
      mscr_crt   fold0    nan   0.00   0.00   0.40   0.86    0     8
      mscr_crt   fold1   1.00   0.12   0.22   0.49   0.89    0    10
      mscr_crt   fold2   1.00   0.25   0.40   0.44   1.00    0     9
  mscr_rowperm  pooled   0.57   1.00   0.73   0.79   0.91    0    24
     shap_gbdt  pooled   1.00   0.25   0.40   0.49   1.00    0    11
     shap_gbdt   fold0   1.00   0.38   0.55   0.60   1.00    1    14
     shap_gbdt   fold1   1.00   0.50   0.67   0.63   1.00    1    15
     shap_gbdt   fold2   1.00   0.50   0.67   0.63   1.00    1    15
          corr  pooled   1.00   0.50   0.67   0.65   0.93    1    17
          corr   fold0   1.00   0.50   0.67   0.65   1.00    1    18
          corr   fold1   1.00   0.50   0.67   0.74   0.94    4    21
          corr   fold2   1.00   0.50   0.67   0.70   0.88    5    21
       granger  pooled   0.75   0.75   0.75   0.70   0.79    4    31
       granger   fold0   1.00   0.38   0.55   0.52   1.00    1    17
       granger   fold1   1.00   0.38   0.55   0.52   1.00    0    15
       granger   fold2   1.00   0.25   0.40   0.56   0.75    2    17
    granger_by  pooled   1.00   0.50   0.67   0.63   0.87    3    23
    granger_by   fold0   1.00   0.25   0.40   0.49   1.00    1    14
    granger_by   fold1   1.00   0.25   0.40   0.46   1.00    0    12
    granger_by   fold2   1.00   0.25   0.40   0.45   0.89    1    13
     two_tower  pooled    nan   0.00   0.00   0.29   1.00    3    10
     two_tower   fold0    nan   0.00   0.00   0.29   1.00    3    10
     two_tower   fold1   1.00   0.12   0.22   0.39   0.86    4    10
     two_tower   fold2    nan   0.00   0.00   0.37   0.86    3    14
           int  pooled   1.00   0.25   0.40   0.44   1.00    0     9
           int   fold0   1.00   0.12   0.22   0.49   0.89    2    11
           int   fold1   1.00   0.38   0.55   0.56   1.00    1    14
           int   fold2   1.00   0.38   0.55   0.56   1.00    1    14
          qacm  pooled    nan   0.00   0.00   0.49   1.00    0    11
          qacm   fold0    nan   0.00   0.00   0.60   1.00    0    14
          qacm   fold1    nan   0.00   0.00   0.53   1.00    0    13
          qacm   fold2    nan   0.00   0.00   0.53   1.00    0    12
P1 FAIL {'indirect_recall': False, 'indirect_precision': True, 'overall_f1': True, 'sign_accuracy': True, 'far_declarations': True}
```
