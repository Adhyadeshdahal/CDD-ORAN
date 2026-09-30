# Step 2 DEV (P3 surge-L40, 40 DEV seeds 184200-184239, Kaggle e6p-s2dev-1, 400 jobs, 7.4 CPU-h)

Candidate referee rules from the step-2 debate (F/G). Not frozen, DEV only. R = Gate A v2 definition (den 47.05).
Eligible = energy retention >= .90 and every guard <= 1.10 x AA. M2 LL guard = builder-defined (G never implemented one).

```
== E6-P step-2 DEV (P3 surge-L40): 40 seeds with records, 40 complete over 10 arms
   V_AA 157.48  V_ref 110.43 (sub:SliceGuarantee)  den 47.05 psvr (Gate A v2 den 45.82)  n 40
   arm                      n       V       R          R 90% CI  R-R(B1)            90% CI   Rfix    ret   svr  nonp    ll   rlf elig  cpu_s
   freeze                  40  437.36  -5.948 [-8.580,-4.368]   -6.246 [-8.913,-4.638] -6.108  0.000  0.59  0.51  0.28  0.77  nE    62.7
   sub:ES+PowerES          40  766.32 -12.939 [-17.298,-10.403]  -13.237 [-17.635,-10.631] -13.287  1.000  1.03  0.88  0.50  1.00  Y    62.5
   noarb                   40  157.48  +0.000 [+0.000,+0.000]   -0.298 [-0.615,+0.016] +0.000  1.006  1.00  1.00  1.00  1.00  Y    48.9
   sub:ES                  40  547.25  -8.283 [-11.364,-6.446]   -8.581 [-11.703,-6.705] -8.506  0.447  0.75  0.65  0.33  0.82  nE    49.2
   sub:PowerES             40  638.03 -10.213 [-14.239,-7.819]  -10.511 [-14.542,-8.079] -10.487  0.675  0.86  0.73  0.50  0.90  nE    64.9
   sub:SliceGuarantee      40  110.43  +1.000 [+1.000,+1.000]   +0.702 [+0.385,+1.016] +1.027 -0.001  0.59  0.58  0.56  0.77  nE    64.9
   B1                      40  143.48  +0.298 [-0.016,+0.615]   +0.000 [+0.000,+0.000] +0.306  1.441  0.99  0.99  1.05  1.27  nG    80.4
   B2                      40  140.50  +0.361 [+0.043,+0.687]   +0.063 [+0.031,+0.107] +0.371  1.402  0.97  0.97  1.03  1.27  nG    80.1
   M1                      40  152.86  +0.098 [-0.068,+0.269]   -0.199 [-0.493,+0.094] +0.101  1.138  1.00  1.00  1.01  1.01  Y    78.3
   M2                      40  152.66  +0.102 [+0.017,+0.188]   -0.195 [-0.511,+0.116] +0.105  1.024  1.01  1.01  1.02  1.03  Y    78.2
   eligible = retention >= 0.9 and every guard ratio <= 1.1 (E = energy fail, G = guard fail); CPU 7.44 h total, 67.0 s/job
```

Reading: blanket PowerES ptx-up rejection (B1/B2) recovers ~.30-.36 but breaks the RLF guard (1.27x);
the map-informed saturated-cell gate (M1/M2) is eligible but recovers only ~.10. No eligible arm near .35.
