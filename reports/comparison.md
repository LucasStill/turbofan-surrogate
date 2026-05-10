# Surrogate sweep comparison

Comparing 4 runs:
  - **tiny** : `runs/sweep/tiny`
  - **small** : `runs/sweep/small`
  - **default** : `runs/sweep/default`
  - **large** : `runs/sweep/large`

## 1. Headline

| Variant | hidden | depth | params | epochs | best val loss | train s | nMAE HPC_Tout | nMAE HP_Nmech | nMAE HPC_Tin | nMAE LPT_Tin | nMAE Fuel_flow | nMAE HPC_Pout_st | nMAE LP_Nmech |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 128 | 2 | 19,719 | 100 | 0.00046 | 15 | 0.0131 | 0.0298 | 0.0116 | 0.0146 | 0.0060 | 0.0053 | 0.0149 |
| small | 256 | 3 | 137,991 | 100 | 0.00031 | 24 | 0.0110 | 0.0242 | 0.0095 | 0.0110 | 0.0043 | 0.0039 | 0.0122 |
| default | 384 | 4 | 453,127 | 100 | 0.00008 | 47 | 0.0053 | 0.0102 | 0.0044 | 0.0056 | 0.0028 | 0.0028 | 0.0073 |
| large | 512 | 5 | 1,063,431 | 100 | 0.00004 | 74 | 0.0040 | 0.0057 | 0.0035 | 0.0042 | 0.0021 | 0.0020 | 0.0059 |

## 2. Per sensor R²

| Sensor | tiny | small | default | large |
|---|---|---|---|---|
| HPC_Tout | 0.9997 | 0.9998 | 0.9999 | 1.0000 |
| HP_Nmech | 0.9983 | 0.9988 | 0.9998 | 0.9999 |
| HPC_Tin | 0.9998 | 0.9998 | 1.0000 | 1.0000 |
| LPT_Tin | 0.9996 | 0.9998 | 0.9999 | 1.0000 |
| Fuel_flow | 0.9999 | 1.0000 | 1.0000 | 1.0000 |
| HPC_Pout_st | 0.9999 | 1.0000 | 1.0000 | 1.0000 |
| LP_Nmech | 0.9996 | 0.9997 | 0.9999 | 0.9999 |

## 3. Per sensor nMAE (×σ, lower is better)

| Sensor | tiny | small | default | large |
|---|---|---|---|---|
| HPC_Tout | 0.0131 | 0.0110 | 0.0053 | 0.0040 |
| HP_Nmech | 0.0298 | 0.0242 | 0.0102 | 0.0057 |
| HPC_Tin | 0.0116 | 0.0095 | 0.0044 | 0.0035 |
| LPT_Tin | 0.0146 | 0.0110 | 0.0056 | 0.0042 |
| Fuel_flow | 0.0060 | 0.0043 | 0.0028 | 0.0021 |
| HPC_Pout_st | 0.0053 | 0.0039 | 0.0028 | 0.0020 |
| LP_Nmech | 0.0149 | 0.0122 | 0.0073 | 0.0059 |

## 4. Throughput (samples/sec, GPU)

| Batch size | tiny | small | default | large |
|---|---|---|---|---|
| 1 | 1502 | 1756 | 1127 | 1067 |
| 256 | 362677 | 415609 | 257499 | 247446 |
| 4096 | 3574082 | 3751369 | 2304138 | 1905447 |
| 16384 | 6096651 | 6120835 | 3766311 | 2994318 |

## 5. Quick takeaway

- Best mean nMAE: **large** (0.0039, 1,063,431 params)
- Smallest within 10% of best: **large** (1,063,431 params)
