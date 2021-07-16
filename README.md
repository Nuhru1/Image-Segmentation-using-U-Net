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

- After extraction, the size of each image is: **1024x768**   
This size is large so we might run to **memory allocation problem** when training our model.
For that we decided to devide each image and its corresponding mask by **256x256**
**Code:**


```
import cv2
import os
base_path = '<path to the images folder>'

for filename in os.listdir(base_path):
    img = cv2.imread(base_path + filename)
    name = filename.split('.')[0]
    for r in range(0,img.shape[0],256):
        for c in range(0,img.shape[1], 256):
            cv2.imwrite(f'<path to the new folder>\\{name}-{r}_{c}.png', img[r:r+256, c:c+256,:])
```

result shown on below picture.

![Screenshot 2021-04-26 144538.png](/.attachments/Screenshot%202021-04-26%20144538-879a7a47-c4da-4d6a-977b-a4a27cc72263.png)

![Screenshot 2021-04-26 144513.png](/.attachments/Screenshot%202021-04-26%20144513-29729fee-6d82-4e25-a11d-067cfd6ee669.png)



 - **Data transformation:**
As Deep learning model expect the images to be in **array** format, so let transform our training images and masks in numpy array format and append all the images arrays in a list. it will be the same process for the masks images.
**Code**:


```
import tensorflow as tf
import cv2
from PIL import Image
from matplotlib import pyplot as plt


image_directory = '<path to images folder>'
mask_directory =  '<path to masks folder>'


SIZE = 256
image_dataset = [] 
mask_dataset = [] 

images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    image = cv2.imread(image_directory+image_name, 0)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    image_dataset.append(np.array(image))



masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    image = cv2.imread(mask_directory+image_name, 0)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    mask_dataset.append(np.array(image))
```

- The code below will **normalize our images** arrays and **rescale the masks** between 0 and 1, because the masks images are black and white format (0~255), we don't need to normalize, we just need to rescale (0~1) it by dividing by 255:

```
#Normalize images
image_dataset = np.expand_dims(tf.keras.utils.normalize(np.array(image_dataset), axis=1),3)
# for masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
```

 - **Splitting dataset into training and testing data**:
90% for training data and 10% for validation data

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)
```








