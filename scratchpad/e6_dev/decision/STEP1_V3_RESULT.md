# Step 1 v3 result (protocol v3 1107dc99 = MSCR-CRT v2 at n 1200): P1v2 PASS -> PARTIAL pending P2v2

Fresh eval_v3 1200 eps (Kaggle e6p-disc-ev3-1), GT gtx-3, K0 placebo plx-c2. Pass 1 (no baselines), local, 74 s,
peak 2.3 GB. JSON: step1_v3_pass1.json. THIRD attempt (v1 KILL, v2 KILL near miss) - report all three.

```
K0 placebo (v2): units 2440, tested 60/60, rate 0.033 (2 <= alpha, k_max 7), BY 0 [] -> PASS [0.6 s]
K1 support: sleep units 4727 rejects 1405 -> PASS; units {'carrier': 26610, 'sleep': 4727, 'ptx': 25355, 'prot_min': 105259}
MSCR-CRT v2 pooled: declared 28/60 (floor ok True) [26.0 s]; folds [20, 21, 20, 23]
   declared ('carrier', 'own', 'pv', -1, 0.0001, -12.629, -43.81)
   declared ('carrier', 'own', 'v', -1, 0.0001, -157.174, -63.31)
   declared ('carrier', 'own', 'e', 1, 0.0001, 7288.309, 107.66)
   declared ('carrier', 'own', 'load', -1, 0.0001, -29.776, -7.9)
   declared ('carrier', 'nbr', 'pv', 1, 0.0001, 4.117, 5.06)
   declared ('carrier', 'nbr', 'v', 1, 0.0001, 44.484, 5.59)
   declared ('carrier', 'far', 'pv', 1, 0.0001, 3.109, 3.98)
   declared ('carrier', 'far', 'v', 1, 0.0001, 37.066, 4.99)
   declared ('sleep', 'own', 'pv', -1, 0.0001, -2.252, -6.23)
   declared ('sleep', 'own', 'v', -1, 0.0001, -25.715, -10.84)
   declared ('sleep', 'own', 'e', -1, 0.0001, -140.008, -54.06)
   declared ('sleep', 'own', 'rlf', -1, 0.0001, -0.107, -10.91)
   declared ('sleep', 'own', 'load', -1, 0.0001, -692.547, -44.1)
   declared ('sleep', 'nbr', 'pv', 1, 0.0001, 10.267, 6.11)
   declared ('sleep', 'nbr', 'v', 1, 0.0001, 257.964, 14.8)
   declared ('sleep', 'nbr', 'e', 1, 0.0001, 2534.114, 5.66)
   declared ('sleep', 'nbr', 'load', 1, 0.0001, 593.097, 29.53)
   declared ('sleep', 'far', 'load', 1, 0.0001, 99.022, 6.53)
   declared ('ptx', 'own', 'pv', 1, 0.0001, 3.408, 12.43)
   declared ('ptx', 'own', 'v', 1, 0.0001, 47.245, 20.49)
   declared ('ptx', 'own', 'e', 1, 0.0001, 4050.168, 60.8)
   declared ('ptx', 'own', 'load', 1, 0.0001, 284.571, 72.2)
   declared ('ptx', 'nbr', 'v', -1, 0.0005, -29.983, -3.5)
   declared ('ptx', 'nbr', 'load', -1, 0.0001, -182.462, -21.25)
   declared ('ptx', 'far', 'v', -1, 0.0015, -24.904, -3.23)
   declared ('ptx', 'far', 'e', -1, 0.0004, -807.122, -3.52)
   declared ('ptx', 'far', 'load', -1, 0.0001, -102.021, -12.46)
   declared ('prot_min', 'own', 'pv', -1, 0.0001, -5.421, -37.35)
GT (dir): {'TRUE': 28, 'NULL': 19, 'INDET': 13}; G premise PASS (sleep->nbr pv mean 9.07 CI [3.34, 15.8]); nbr TRUE [('carrier', 'pv', 1), ('ptx', 'e', -1), ('ptx', 'load', -1), ('ptx', 'v', -1), ('sleep', 'e', 1), ('sleep', 'load', 1), ('sleep', 'pv', 1), ('sleep', 'v', 1)]

        method   split  ind P  ind R ind F1   ov P     F1   sign  far  #dec  plc
   mscr_crt_v2  pooled   1.00   0.88   0.93   0.92   0.87   1.00    6    28    0
   mscr_crt_v2   fold0   1.00   0.50   0.67   0.94   0.74   1.00    2    20     
   mscr_crt_v2   fold1   1.00   0.50   0.67   0.95   0.77   1.00    2    21     
   mscr_crt_v2   fold2   1.00   0.38   0.55   0.94   0.74   1.00    2    20     
   mscr_crt_v2   fold3   1.00   0.75   0.86   0.95   0.79   1.00    3    23     
chain set C (pooled MSCR-CRT v2):
   sleep|nbr|load     exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 593.1 z 29.53
   sleep|nbr|pv       exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 10.27 z 6.11
   sleep|nbr|v        exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 258 z 14.80
   ptx|nbr|load       exp -1 GT TRUE(-1) declared True sign -1 p 0.0001 beta -182.5 z -21.25
P1v2 PASS {'premise_edge_declared_plus': True, 'chain_hits': True, 'overall_precision': True, 'sign_accuracy': True} values {'chain_true': 4, 'chain_need': 3, 'chain_hits': 4, 'overall_precision': 0.92, 'sign_accuracy': 1.0, 'far_declared (reported, not a criterion)': 6, 'indirect_recall_all_gt_true_nbr': 0.875}
P2v2 not evaluated (--no-baselines)
placebo declarations (validity): {'mscr_crt_v2': {'placebo': 0}}
VERDICT: PARTIAL [P2v2 not evaluated: --no-baselines]
timing (s): {'k0_placebo': 0.6, 'unit_tables': 17.3, 'crt_v2_pooled': 26.0, 'crt_v2_fold0': 5.9, 'crt_v2_fold1': 6.1, 'crt_v2_fold2': 6.1, 'crt_v2_fold3': 6.0, 'total': 73.5}; peak RSS 2251.8 MB
```

## Pass 2 (baselines; v1 code path + DEV far-FPR tau; local): P2v2 FAIL (folds 1/4) -> FINAL VERDICT PARTIAL

Pooled: MSCR v2 indirect F1 .93 vs best baseline .67 (corr). Per 300-episode fold, the best baseline beats MSCR in folds 0-2
(Granger(-BY) .71-.75 vs .67 in folds 0-1; |corr| .67 vs .55 in fold 2; MSCR wins fold 3 at .86) with overall precision .61-.71 and 9-13 placebo declarations.

```
        method   split  ind P  ind R ind F1   ov P     F1   sign  far  #dec  plc
   mscr_crt_v2  pooled   1.00   0.88   0.93   0.92   0.87   1.00    6    28    0
   mscr_crt_v2   fold0   1.00   0.50   0.67   0.94   0.74   1.00    2    20     
   mscr_crt_v2   fold1   1.00   0.50   0.67   0.95   0.77   1.00    2    21     
   mscr_crt_v2   fold2   1.00   0.38   0.55   0.94   0.74   1.00    2    20     
   mscr_crt_v2   fold3   1.00   0.75   0.86   0.95   0.79   1.00    3    23     
     shap_gbdt  pooled   1.00   0.25   0.40   0.89   0.43   1.00    0     9    6
     shap_gbdt   fold0   1.00   0.25   0.40   0.90   0.47   1.00    0    11     
     shap_gbdt   fold1   1.00   0.25   0.40   0.90   0.47   1.00    0    10     
     shap_gbdt   fold2   1.00   0.25   0.40   0.90   0.47   1.00    0    10     
     shap_gbdt   fold3   1.00   0.25   0.40   0.90   0.47   1.00    0    10     
          corr  pooled   1.00   0.50   0.67   0.86   0.57   1.00    0    16   16
          corr   fold0   1.00   0.50   0.67   0.86   0.57   1.00    0    16     
          corr   fold1   1.00   0.50   0.67   0.86   0.57   1.00    0    16     
          corr   fold2   1.00   0.50   0.67   0.86   0.57   1.00    0    16     
          corr   fold3   1.00   0.38   0.55   0.86   0.57   1.00    1    16     
       granger  pooled   0.55   0.75   0.63   0.61   0.70   0.78   10    42   13
       granger   fold0   0.67   0.75   0.71   0.70   0.72   0.81    5    35     
       granger   fold1   0.75   0.75   0.75   0.66   0.67   0.79    6    33     
       granger   fold2   0.56   0.62   0.59   0.69   0.73   0.77    7    36     
       granger   fold3   0.67   0.75   0.71   0.62   0.67   0.80    6    34     
    granger_by  pooled   0.55   0.75   0.63   0.62   0.71   0.78    9    41    9
    granger_by   fold0   0.75   0.75   0.75   0.71   0.71   0.80    4    30     
    granger_by   fold1   0.71   0.62   0.67   0.69   0.67   0.83    4    28     
    granger_by   fold2   0.62   0.62   0.62   0.69   0.70   0.80    6    31     
    granger_by   fold3   0.67   0.75   0.71   0.65   0.68   0.80    6    33     
     two_tower  pooled    nan   0.00   0.00   0.57   0.23   1.00    3     8   10
     two_tower   fold0    nan   0.00   0.00   0.64   0.36   1.00    3    13     
     two_tower   fold1    nan   0.00   0.00   0.67   0.24   1.00    2     7     
     two_tower   fold2   1.00   0.12   0.22   0.70   0.37   0.86    2    11     
     two_tower   fold3   0.00   0.00   0.00   0.58   0.35   1.00    3    15     
           int  pooled   1.00   0.25   0.40   0.86   0.34   1.00    0     7    6
           int   fold0   1.00   0.12   0.22   0.83   0.29   1.00    0     6     
           int   fold1   1.00   0.25   0.40   0.86   0.34   1.00    0     7     
           int   fold2   1.00   0.25   0.40   0.86   0.34   1.00    0     7     
           int   fold3   1.00   0.25   0.40   0.86   0.34   1.00    0     7     
          qacm  pooled    nan   0.00   0.00   0.88   0.39   1.00    0     8   11
          qacm   fold0    nan   0.00   0.00   0.90   0.47   1.00    0    10     
          qacm   fold1    nan   0.00   0.00   0.90   0.47   1.00    0    10     
          qacm   fold2    nan   0.00   0.00   0.88   0.39   1.00    0     9     
          qacm   fold3    nan   0.00   0.00   0.86   0.34   1.00    0     7     
chain set C (pooled MSCR-CRT v2):
   sleep|nbr|load     exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 593.1 z 29.53
   sleep|nbr|pv       exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 10.27 z 6.11
   sleep|nbr|v        exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 258 z 14.80
   ptx|nbr|load       exp -1 GT TRUE(-1) declared True sign -1 p 0.0001 beta -182.5 z -21.25
P1v2 PASS {'premise_edge_declared_plus': True, 'chain_hits': True, 'overall_precision': True, 'sign_accuracy': True} values {'chain_true': 4, 'chain_need': 3, 'chain_hits': 4, 'overall_precision': 0.92, 'sign_accuracy': 1.0, 'far_declared (reported, not a criterion)': 6, 'indirect_recall_all_gt_true_nbr': 0.875}
P2v2 FAIL (folds ok 1/4); pooled MSCR v2 0.9333333333333333 vs best 0.6666666666666666 (corr)
placebo declarations (validity): {'mscr_crt_v2': {'placebo': 0}, 'shap_gbdt': {'placebo': 6}, 'corr': {'placebo': 16}, 'granger': {'placebo': 13}, 'granger_by': {'placebo': 9}, 'two_tower': {'placebo': 10}, 'int': {'placebo': 6}, 'qacm': {'placebo': 11}}
VERDICT: PARTIAL
timing (s): {'k0_placebo': 0.5, 'unit_tables': 16.7, 'crt_v2_pooled': 27.4, 'crt_v2_fold0': 6.9, 'crt_v2_fold1': 6.9, 'crt_v2_fold2': 6.4, 'crt_v2_fold3': 6.1, 'baselines': 3347.6, 'total': 3424.1}; peak RSS 2935.1 MB
placebo declarations (validity): {'mscr_crt_v2': {'placebo': 0}, 'shap_gbdt': {'placebo': 6}, 'corr': {'placebo': 16}, 'granger': {'placebo': 13}, 'granger_by': {'placebo': 9}, 'two_tower': {'placebo': 10}, 'int': {'placebo': 6}, 'qacm': {'placebo': 11}}
```
