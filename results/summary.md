# Benchmark summary

Cases: 4 images x 3 PSF x 3 noise levels = 36

## Mean PSNR / SSIM by noise level

|   noise_sigma | method                     |   psnr |   ssim |
|--------------:|:---------------------------|-------:|-------:|
|          0.01 | RL + TV + discrepancy stop | 29.334 |  0.806 |
|          0.01 | RL + discrepancy stop      | 28.472 |  0.779 |
|          0.01 | RL, 30 iter                | 28.207 |  0.733 |
|          0.01 | RL, oracle iters           | 28.889 |  0.745 |
|          0.01 | Wiener + power-law prior   | 29.326 |  0.76  |
|          0.01 | Wiener, K=1e-2             | 27.002 |  0.615 |
|          0.01 | Wiener, oracle K           | 27.557 |  0.66  |
|          0.01 | blurred (no restoration)   | 25.899 |  0.692 |
|          0.05 | RL + TV + discrepancy stop | 26.4   |  0.687 |
|          0.05 | RL + discrepancy stop      | 26.26  |  0.672 |
|          0.05 | RL, 30 iter                | 21.554 |  0.316 |
|          0.05 | RL, oracle iters           | 26.446 |  0.637 |
|          0.05 | Wiener + power-law prior   | 26.932 |  0.683 |
|          0.05 | Wiener, K=1e-2             | 17.817 |  0.196 |
|          0.05 | Wiener, oracle K           | 23.037 |  0.454 |
|          0.05 | blurred (no restoration)   | 22.394 |  0.318 |
|          0.1  | RL + TV + discrepancy stop | 25.153 |  0.601 |
|          0.1  | RL + discrepancy stop      | 25.098 |  0.592 |
|          0.1  | RL, 30 iter                | 16.981 |  0.167 |
|          0.1  | RL, oracle iters           | 25.168 |  0.565 |
|          0.1  | Wiener + power-law prior   | 25.888 |  0.648 |
|          0.1  | Wiener, K=1e-2             | 13.017 |  0.095 |
|          0.1  | Wiener, oracle K           | 20.82  |  0.334 |
|          0.1  | blurred (no restoration)   | 18.799 |  0.16  |


## Best blind modification vs best classical method with default params (PSNR, dB)

- mean +5.23 dB, median +3.47 dB, wins 36/36

## Best blind modification vs ORACLE-tuned classical methods (uses ground truth)

- mean +0.65 dB, median +0.47 dB, wins 33/36

By noise level (vs oracle):

|   noise_sigma |   mean |   median |
|--------------:|-------:|---------:|
|          0.01 |   0.74 |     0.73 |
|          0.05 |   0.49 |     0.23 |
|          0.1  |   0.72 |     0.4  |

## Median RL iterations chosen by discrepancy principle

|   noise_sigma |   iters |
|--------------:|--------:|
|          0.01 |    30.5 |
|          0.05 |     4   |
|          0.1  |     2   |