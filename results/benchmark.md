| Scenario | PID | LQR | Geometric | MPC |
|---|---:|---:|---:|---:|
| `hover_recovery` | 17.74 (1.38 s) | 19.57 (0.97 s) | 19.77 (0.91 s) | 20.28 (1.01 s) |
| `step` | 40.51 (1.30 s) | 43.98 (1.22 s) | 43.17 (1.02 s) | 44.23 (1.25 s) |
| `long_step` | 92.56 (0.81 s) | crash | 92.79 (1.18 s) | 102.01 (1.65 s) |
| `circle` | 0.93 | <0.01 | <0.01 | <0.01 |
| `figure8` | 4.33 | 0.04 | 0.03 | 0.04 |
| `figure8_fast` | 9.29 | 0.23 | 0.10 | 0.27 |
| `flip_recovery` | 8.14 (1.80 s) | 9.48 (1.05 s) | 23.11 (1.39 s) | 8.12 (1.03 s) |
| `mass_mismatch` | 10.16 | 20.85 | 19.51 | 22.32 |
| `motor_lag` | 4.52 | 0.65 | 0.47 | 0.68 |
| `wind` | 2.25 | 7.16 | 5.73 | 7.65 |
| `sensor_noise` | 4.34 | 0.16 | 0.15 | 0.17 |

Mean controller compute time per call (Python, single core): PID 0.16 ms, LQR 0.84 ms, Geometric 0.96 ms, MPC 3.31 ms.
