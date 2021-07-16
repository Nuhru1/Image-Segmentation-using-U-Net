# Image-Segmentation-using-U-Net

## Goal
Try to accurately segment mitochondria and synapses.

## Dataset
The dataset available for download on this [link](https://www.epfl.ch/labs/cvlab/data/data-em/) represents a **5x5x5µm** section taken from the CA1 hippocampus region of the brain, corresponding to a **1065x2048x1536** volume. The resolution of each voxel is approximately **5x5x5nm**. The data is provided as **multipage TIF files** that can be loaded in Fiji.

## Data preparation
The dataset in the original website contains **2 sub-volumes**. Each sub-volume consists of the first **165** slices of the **1065x2048x1536** image stack.
In this work we only used **1 sub-volumes** which contains the **training sub-volume** and the **groundtruth training sub-volume**

![Screenshot 2021-07-16 122020](https://user-images.githubusercontent.com/44145876/125890905-f36c133c-87e0-4d48-bbc2-b14a7f957015.png)


- **Extract pages from TIF file:**
    - **What is TIF files:**
A TIF file contains an image saved in the **Tagged Image File Format ( TIFF )**, a high-quality graphics format. It is often used for storing images with many colors, typically digital photos, and includes support for **layers** and **multiple pages.**


![Screenshot 2021-04-26 143352.png](/.attachments/Screenshot%202021-04-26%20143352-dd85afde-0011-4870-9012-499f68a46abe.png)

   The code below will extract the **pages (images)** from the tif file.

```
from PIL import Image, ImageSequence
im = Image.open("<path to the tif file>")
for i, page in enumerate(ImageSequence.Iterator(im)):
    page.save("image%d.png" % i)

```
