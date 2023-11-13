# Exploring-Untrained-NNs-Project-ETHZ-2023-
Semester project on understanding training free neural networks

### Exploring Deep Decoder applied to MRI

This repository provides code to reproduce the results from the semester project named as ‘Exploring Deep Decoder applied to MRI’ by Pablo Blasco Fernándes. Advised by Jakob Geusen and Gustav Bredell and supervised by Ender Konukoglu

Untrained neural networks have emerged as a potential solution to address the scarcity of large
datasets in deep learning, as they do not rely on previous learning. This characteristic can prove
valuable in domains where a trade-off exists between training and test data or when labeled data
is not readily available. In this project, we explore the capabilities of the untrained model called
the Deep Decoder.
Our analysis involves introducing variations in the model’s settings to gain insights into their
impact on performance, specifically focusing on inference time, stability and the quality of its
outputs so that one could asses how useful this model can be for further specific tasks.
After using variations of the inputs and targets of the model, and specifically by using MRI
slices, we analyze how different hyperparameters play a role. Our research found that indeed
choosing the set of hyperparameters conditions the robustness of the model. Understanding these
relationships can aid in optimizing the model’s performance and guiding its application

Please install following packages before usage.

```
conda create --name DD python=3.8.5
conda activate DD
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install scikit-image scikit-learn matplotlib pandas
pip install piq
pip install noise
pip install nibabel
pip install opencv-python
```

Each folder comprises the code used in the set of experiments if the project. The .py files in each folder can be run to obtain the .csv files for the grid search. The Jupyter Notebooks .ipynb have the same code but allow to interact with the output 
