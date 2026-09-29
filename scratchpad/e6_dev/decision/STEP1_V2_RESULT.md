# Step 1 v2 result (frozen protocol v2 fea6488c): KILL (P1v2 fails on the premise edge only)

Fresh seeds: eval_v2 480 eps (Kaggle e6p-disc-ev2-2), gt_ext 20 eps (e6p-disc-gtx-3), placebo_ext 20 eps (Colab
e6p-disc-plx-c2). Pass 1 (no baselines), local, 29 s, peak 1 GB. JSON: step1_v2_pass1.json. P2v2 (baselines) runs
separately and cannot change the verdict (P1v2 already fails).

```
K0 placebo (v2): units 2440, tested 60/60, rate 0.033 (2 <= alpha, k_max 7), BY 0 [] -> PASS [0.5 s]
K0 placebo-dev (descriptive): rate 0.067, BY 0
K1 support: sleep units 1912 rejects 611 -> PASS; units {'carrier': 10533, 'sleep': 1912, 'ptx': 10051, 'prot_min': 42133}
MSCR-CRT v2 pooled: declared 23/60 (floor ok True) [9.8 s]; folds [19, 18, 19, 17]
   declared ('carrier', 'own', 'pv', -1, 0.0001, -12.054, -26.5)
   declared ('carrier', 'own', 'v', -1, 0.0001, -155.772, -40.17)
   declared ('carrier', 'own', 'e', 1, 0.0001, 6992.777, 66.7)
   declared ('carrier', 'own', 'load', -1, 0.0001, -39.616, -6.87)
   declared ('carrier', 'far', 'pv', 1, 0.0002, 4.382, 3.48)
   declared ('sleep', 'own', 'pv', -1, 0.0001, -2.674, -5.06)
   declared ('sleep', 'own', 'v', -1, 0.0001, -28.121, -7.33)
   declared ('sleep', 'own', 'e', -1, 0.0001, -143.625, -35.76)
   declared ('sleep', 'own', 'rlf', -1, 0.0001, -0.101, -7.05)
   declared ('sleep', 'own', 'load', -1, 0.0001, -680.71, -29.44)
   declared ('sleep', 'nbr', 'v', 1, 0.0001, 272.25, 9.56)
   declared ('sleep', 'nbr', 'e', 1, 0.0002, 2857.577, 4.06)
   declared ('sleep', 'nbr', 'load', 1, 0.0001, 562.113, 19.81)
   declared ('sleep', 'far', 'v', -1, 0.0005, -73.245, -3.37)
   declared ('sleep', 'far', 'load', 1, 0.0001, 117.046, 5.34)
   declared ('ptx', 'own', 'pv', 1, 0.0001, 3.337, 7.6)
   declared ('ptx', 'own', 'v', 1, 0.0001, 41.118, 11.33)
   declared ('ptx', 'own', 'e', 1, 0.0001, 3940.927, 36.74)
   declared ('ptx', 'own', 'load', 1, 0.0001, 277.561, 44.67)
   declared ('ptx', 'nbr', 'load', -1, 0.0001, -162.307, -11.66)
   declared ('ptx', 'far', 'v', -1, 0.0003, -51.306, -4.12)
   declared ('ptx', 'far', 'load', -1, 0.0001, -114.786, -8.59)
   declared ('prot_min', 'own', 'pv', -1, 0.0001, -5.969, -25.17)
GT (dir): {'TRUE': 28, 'NULL': 19, 'INDET': 13}; G premise PASS (sleep->nbr pv mean 9.07 CI [3.34, 15.8]); nbr TRUE [('carrier', 'pv', 1), ('ptx', 'e', -1), ('ptx', 'load', -1), ('ptx', 'v', -1), ('sleep', 'e', 1), ('sleep', 'load', 1), ('sleep', 'pv', 1), ('sleep', 'v', 1)]

        method   split  ind P  ind R ind F1   ov P     F1   sign  far  #dec  plc
   mscr_crt_v2  pooled   1.00   0.50   0.67   0.95   0.82   0.95    5    23    0
   mscr_crt_v2   fold0   1.00   0.50   0.67   0.94   0.74   1.00    2    19     
   mscr_crt_v2   fold1   1.00   0.38   0.55   0.94   0.71   0.94    2    18     
   mscr_crt_v2   fold2   1.00   0.38   0.55   0.94   0.74   1.00    2    19     
   mscr_crt_v2   fold3   1.00   0.38   0.55   0.94   0.68   1.00    1    17     
chain set C (pooled MSCR-CRT v2):
   sleep|nbr|load     exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 562.1 z 19.81
   sleep|nbr|pv       exp +1 GT TRUE(+1) declared False sign +1 p 0.007 beta 6.972 z 2.67
   sleep|nbr|v        exp +1 GT TRUE(+1) declared True sign +1 p 0.0001 beta 272.3 z 9.56
   ptx|nbr|load       exp -1 GT TRUE(-1) declared True sign -1 p 0.0001 beta -162.3 z -11.66
P1v2 FAIL {'premise_edge_declared_plus': False, 'chain_hits': True, 'overall_precision': True, 'sign_accuracy': True} values {'chain_true': 4, 'chain_need': 3, 'chain_hits': 3, 'overall_precision': 0.9523809523809523, 'sign_accuracy': 0.95, 'far_declared (reported, not a criterion)': 5, 'indirect_recall_all_gt_true_nbr': 0.5}
P2v2 not evaluated (--no-baselines)
placebo declarations (validity): {'mscr_crt_v2': {'placebo': 0, 'placebo_dev': 0}}
VERDICT: KILL [P2v2 not evaluated: --no-baselines]
timing (s): {'k0_placebo': 0.5, 'unit_tables': 6.6, 'crt_v2_pooled': 9.8, 'crt_v2_fold0': 2.4, 'crt_v2_fold1': 2.4, 'crt_v2_fold2': 2.4, 'crt_v2_fold3': 2.4, 'total': 29.3}; peak RSS 989.3 MB
```
