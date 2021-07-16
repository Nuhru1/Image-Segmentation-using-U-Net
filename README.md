# Image-Segmentation-using-U-Net

## Goal
Try to accurately segment mitochondria and synapses.

## Dataset
The dataset available for download on this [link](https://www.epfl.ch/labs/cvlab/data/data-em/) represents a **5x5x5µm** section taken from the CA1 hippocampus region of the brain, corresponding to a **1065x2048x1536** volume. The resolution of each voxel is approximately **5x5x5nm**. The data is provided as **multipage TIF files** that can be loaded in Fiji.

## Data preparation
The dataset in the original website contains **2 sub-volumes**. Each sub-volume consists of the first **165** slices of the **1065x2048x1536** image stack.
In this work we only used **1 sub-volumes** which contains the **training sub-volume** and the **groundtruth training sub-volume**

![Screenshot 2021-07-16 122020](https://user-images.githubusercontent.com/44145876/125890905-f36c133c-87e0-4d48-bbc2-b14a7f957015.png)
