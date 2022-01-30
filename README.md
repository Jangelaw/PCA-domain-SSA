# PCA-domain-SSA

This code is for [1], and can only be used for non-comercial purpose. If you use our code, please cite [1]. Code Author: Yijun Yan Email: y.yan2@rgu.ac.uk 
Note: some functions are employed from [2] and [3]

[1] Yijun.Yan, Jinchang Ren, et al., PCA-domain Fused Singular Spectral Analysis for fast and Noise-Robust Spectral-Spatial Feature Mining in Hyperspectral Classification. IEEE Geoscience and Remote Sensing Letters, 2021.

[2] Jaime Zabalza, Jinchang Ren, et al., Novel folded-PCA for improved feature extraction and data reduction with hyperspectral imaging and SAR in remote sensing, ISPRS Journal of Photogrammetry and Remote Sensing, 2014

[3] Jaime Zabalza, Jinchang Ren, et al., Novel two-dimensional singular spectrum analysis for effective feature extraction and data classification in hyperspectral imaging, IEEE transactions on geoscience and remote sensing , 2015

# Requirement
Download the libsvm toolbox and move it to the root file
Libsvm: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

# Dataset
Download the dataset and move it to the data file
Salinas and PaviaU: https://rslab.ut.ac.ir/data

# Usage
IP- Indianpines corrected
SA- Salinas corrected
PU- PaviaU
Simply run the P_SSA_IP to get the predicted results of IndianPines corrected data using 5,10,15,20,25,30,5%, and 10% training samples.
