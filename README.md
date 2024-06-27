# Dogs-breed-classification-model
# Dog-Breed-Image-Classification
### Motivation
#### Knowledge Enhancement in Deep Learning 
The purpose of this project is to enhance the fundamental understanding of Convolutional Neural Network, have a handful of experience working with Tensorflow and Keras libraries, and apply transfer learning techniques to see how the model changes. 
#### For Future Dog Owner 
Also, as a future dog parent, understanding the difference between breeds and get familiar with them will be helpful to select the best dog that fits your lifestyle.




### Dataset & Data Preprocessing 
The dataset is from [Stanford University](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html), with a total of 20580 images and 120 breeds. Also, we can do one extra step to help us better understand the difference between dogs, which is to use the American Kennel Club's [List of Breeds by Group](https://www.akc.org/public-education/resources/general-tips-information/dog-breeds-sorted-groups/) information to assign 120 breeds into 9 groups. By doing that, we may able to discover more insights about puppies. 

### Convolutional Neural Network (CNN) Baseline Model

We will try to build a CNN model from scratch and here is the structure of this project. 

The CNN consists of 3 layers:
1. Convolution (filter)
      * Scan through the image to get **features** of an image
      * Try to detect edge & shape 
      * Follow by Activation Map
2. Pooling
      * Get the most important feature
      * Reduce dimensions of data (image)
3. Fully Connected
      * To generate output 
      * Make prediction with associate features

Our final CNN baseline model consists of 5 layers. With a total of 4 filter and pooling layers and 1 fully connected layer.


### Transfer Learning - Inception V3 Model
Inception V3 Model is a pre-trained model on more than a million images, 1,000 classes from the ImageNet database developed by Google. It has stacked layers compare to the traditional CNN model and it is up to 48 layers. V3 here means version 3.

The core idea of Inception V3:
1. Multiple fitter layers with different sizes running at the same level.
2. 1x1 convolution before processing 3v3 or 5v5 filter layers to reduce the dimension of the image.


### CNN Baseline Model Performance VS. Inception V3 Model Performance
#### CNN Model 

<table>
<tr><th>120 Breeds </th><th> 9 Groups </th></tr>
<tr><td>

|Dataset | Accuracy |
| ----------- | -------- |
| Training  | 3.88%|
| Validation  | 3.34% |

</td><td>

|Dataset | Accuracy |
| ----------- | -------- |
| Training  | 28.23%|
| Validation  | 26.24% |

</td></tr> </table>
#### Inception V3 Model

<table>
<tr><th>120 Breeds </th><th> 9 Groups </th></tr>
<tr><td>

|Dataset | Accuracy |
| ----------- | -------- |
| Training  | 75.58%|
| Validation  | 73.00%|

</td><td>

|Dataset | Accuracy |
| ----------- | -------- |
| Training  | 80.75%|
| Validation  | 81.43%|

</td></tr> </table>

### Challenges
1. Small Sample Size (20580 images for 120 breeds – Average 172 Images).
2. Images include many surrounding features that are not helpful to identify the breed.
3. No standard of features. Puppy will grow in different sizes and has different features (eg, puppy ears will grow bigger overtimes).

### Future Improvement
We can always add more functionalities to it, such as identify other animals. If the image is a cat, we will not make any prediction of the breed. We can also develop an interactive application to allow users to upload their images and see which breed they look similar to.

##### Reference
1. [Keras - Image data preprocessing](https://keras.io/api/preprocessing/image/)
2. [COMPUTER VISION - Keras ImageDataGenerator with flow_from_dataframe()](https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_dataframe/)
3. [Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs](http://people.csail.mit.edu/khosla/papers/fgvc2011.pdf)
4. [How to easily build a Dog breed Image classification model](https://medium.com/nanonets/how-to-easily-build-a-dog-breed-image-classification-model-2fd214419cde)

