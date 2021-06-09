# Sine Wave Experiments

This subdirectory contains code to train the LatSegODE on Sine Wave data. This code was used to generate Appendix K, which demonstrates an ablation study on segmentation performance as a function of train set size and the number of samples per training trajectory.


To generate the Sine Wave dataset, call `generate_data.py`. This will generate a training, validation, and test set of sine waves with the follow options:

Flag | Descriptions | Default
------------ | ------------- | -------------
--n_train_traj | Number of trajectories in training set. | 10000
--n_test_traj | Number of trajectories in test set. | 25
--n_train_samp | Number of samples in each training trajectory | 100
--n_test_samp | Number of samples in each test trajectory | 100
--length | Length of each trajectory (timescale) | 7
--noise | Standard deviation of random noise added to trajectories | 0.025
--amp_min | Minimum absolute amplitude change between segments in test set | 3

Other relevant options such as the range of possible amplitudes and frequencies can be changed by modifying the file directly. Defaults are (-10, 10) and (1, 4), respectively. The default dataset can be generated using command:

```python generate_data.py```

LatSegODE models can be trained by calling:

```python train_model.py --data_file {data_file_name}```. 

Model and training hyperparameters can be set by modifying this file. See main README for further description. The data file is assumed to be located at the default output folder of the generation script.

Finally, LatSegODE models can be evaluated by calling:

```python evaluate_model.py --data_file {data_file_name} --model_file {model_file_name}```

This script outputs the RandIndex, Hausdorff Metric, F1 score, and Annotation error on the test set. See the paper for a description of these metrics.
A full description of segmentation parameters can be found in the main README. 
