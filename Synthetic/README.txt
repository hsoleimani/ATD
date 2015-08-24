This file explains how to reproduce the results on synthetic data set.
First, generate synthetic data by running 'datagen.py'.

##### ATD
1. Run 'PyRun.py' in the PTM folder to learn the null topics
2. Run 'ATD.py' to detect canidate clusters from the test set
3. Run 'bootstrap.py' to perform significance test on the detected clusters
4. Run 'print_ATD_results.py' to print ATD results

##### LB Method
1. Run 'LB/LB.py' to run likelihood-based method for group AD.
2. Run 'LB/LB_indv.py' to run likelihood-based method for individual point AD.

##### SVM Method
1. Run 'OneClassSVM/main.py' to run One-Class SVM for group AD.
2. Run 'OneClassSVM/main_indv.py' to run One-Class SVM for individual point AD.

##### NN Method
1. Run 'KNN/main.py' to run Nearest Neighbor method for group AD.
2. Run 'KNN/main_indv.py' to run Nearest Neighbor method individual point AD.

##### M4 Method
Run 'M4/main.py' to run M4 for group AD.
