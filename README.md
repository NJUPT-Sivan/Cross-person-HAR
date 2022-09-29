# Cross-Person HAR

The content of this article was published in IEEE VTC2022-Fall

## Dataset

1. Download dataset from here: 

   Link: https://pan.baidu.com/s/1Tgwd8CIDaWfFCCbfrRX8DQ?pwd=axhz 

2. A dataset for seven different daily human activities including wave, clap, walk, liedown, sitdown, fall and pickup in an indoor environment.

3. We use [ESP32 CSI Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)  to collect CSV files in our dataset.



## Code

1. SE-ABLSTM-trainmodels.py: Train ten models at once with cosine annealing learning rate
2. SE-ABLSTM-test.py: Use previously generated models to make ensemble predictions on datasets of different people



## Reference

1. https://github.com/ermongroup/Wifi_Activity_Recognition

2. https://github.com/parisafm/CSI-HAR-Dataset

3. https://github.com/ludlows/CSI-Activity-Recognition

   









