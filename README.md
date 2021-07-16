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

![Screenshot 2021-07-16 123624](https://user-images.githubusercontent.com/44145876/125892071-19609e15-f99a-4fcd-929f-b5df2ecb7ecd.png)


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


# Building our Model:
Find codes in the repo.

# Evaluation


```
# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
```

![Screenshot 2021-04-26 154647.png](/.attachments/Screenshot%202021-04-26%20154647-97f8f9a2-c01a-457d-89e6-b6d679d556e9.png)




```
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
Result:


![acc.png](/.attachments/acc-f43a8727-8e20-4811-aa81-1b8d9c26b6b3.png)![loss.png](/.attachments/loss-70d49ea4-4c53-489b-94b3-5d33e5dc3205.png)


# Test:

```
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)


plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
```
Result:
![Screenshot 2021-04-26 175919.png](/.attachments/Screenshot%202021-04-26%20175919-5eb83449-7926-47ae-8575-f09ebd0cde4b.png)




# IoU Metric:
In general **accuracy** is not the best metric to evaluate the performance of the model when performing segmentation.
As we know, the accuracy will **divide** the number of **correctly classified pixels** **by** **total number of pixels** on the image.
so, if the background is dominant on the picture, the accuracy value will be high even if the important region is not correctly segmented. 
Example:
In the picture below, the accuracy will be very high because, high number of background pixels has been correctly classified, but the white part which is important is not satisfying.

![Screenshot 2021-04-26 192017.png](/.attachments/Screenshot%202021-04-26%20192017-9dc79382-3eb8-4b9c-bcf7-52800a8bedab.png)

**IoU** is the best metric for segmentation problems.
 - # What is IoU?

The Intersection over Union (**IoU**) metric, also referred to as the **Jaccard index** or **Jaccard coefficient**, is essentially a method to **quantify** the percent **overlap** between the **target mask and our prediction output**.

Simply, the IoU metric measures the number of pixels **common** between the target and prediction masks divided by the total number of pixels present across both masks.

![Screenshot 2021-04-28 104435.png](/.attachments/Screenshot%202021-04-28%20104435-a75292f9-9b63-4114-8358-1c7e981e06e8.png)

**Example:**
let calculate the IoU score of the following **prediction**, given the **ground truth** labeled mask

![Screenshot 2021-04-28 105026.png](/.attachments/Screenshot%202021-04-28%20105026-859c4bdf-ebae-4564-861c-bac37184aca0.png)

The intersection (**A∩B**) is comprised of the pixels found in both the prediction mask and the ground truth mask, whereas the union (**A∪B**) is simply comprised of all pixels found in either the prediction or target mask.

![Screenshot 2021-04-28 105334.png](/.attachments/Screenshot%202021-04-28%20105334-e39ad1eb-5915-4f20-b005-4cca97353ab6.png)


Code:

```
intersection = np.logical_and(target, prediction)
union = np.logical_or(target, prediction)
iou_score = np.sum(intersection) / np.sum(union)
```




# Training our model with **IoU** (Jaccrad coefficient):

Code:

```
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
```

The funtion above is as metricwhen compiling our model:
`model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [jacard_coef])`

#Result:
After 20 epochs, we got: **loss: 0.0116 - jacard_coef: 0.8877 - val_loss: 0.0123 - val_jacard_coef: 0.8772**

![Figure 2021-04-28 164009.png](/.attachments/Figure%202021-04-28%20164009-d747271e-263f-424c-bb32-2325418ec2c2.png)![Figure 2021-04-28 164016.png](/.attachments/Figure%202021-04-28%20164016-a815e8a7-8bf9-4184-bcaf-795d1a6a60ef.png)


Some predictions on testt images:

![Figure 2021-04-28 164118.png](/.attachments/Figure%202021-04-28%20164118-93b4d49d-a6d0-4106-bd44-914483eb0cb4.png)
![Figure 2021-04-28 164132.png](/.attachments/Figure%202021-04-28%20164132-95de1f41-77f8-43ec-bb46-14e223f3320d.png)
![Figure 2021-04-28 164125.png](/.attachments/Figure%202021-04-28%20164125-47be130e-eb9a-4a18-a45d-125afb5f8f81.png)


# Dice Coefficient and Dice loss:

**Dice coefficient**, which is essentially a measure of overlap between two samples. This measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap.

![Screenshot 2021-04-28 173506.png](/.attachments/Screenshot%202021-04-28%20173506-d78bea81-42ba-4f42-894e-40aab7b01cd6.png)

 **|A∩B|** is approximated as the **element-wise multiplication** between the **prediction** and **target** mask, and then sum the resulting matrix.

**Example**:

![Screenshot 2021-04-28 173741.png](/.attachments/Screenshot%202021-04-28%20173741-2d6bd45a-353a-4713-b853-2cd069797b3a.png)


To quantify **|A|** and **|B|**, some researchers use the **simple sum** whereas other researchers prefer to use the **squared sum** for this calculation. 


**1−Dice** is used to formulate the loss function which can be minimized. This loss function is known as the **soft Dice loss** because we directly use the predicted probabilities instead of thresholding and converting them into a binary mask.

![Screenshot 2021-04-28 175534.png](/.attachments/Screenshot%202021-04-28%20175534-cc293b77-779f-4957-b41e-352e4ae5ffb8.png)

Code:


```
def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch
```

h






