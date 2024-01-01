## Contents

### [1. Overview](#overview)

#### [1.1 Problem Statement](#problem-statement)
#### [1.2 Objective](#objective)
#### [1.3 Approach](#approach)

### [2. Data Preprocessing and Data Loading](#data-preprocessing-and-data-loading)

#### [2.1 Replace Duplicate Images](#replace-duplicate-images)
#### [2.2 Replace Wrong Images](#replace-wrong-images)
#### [2.3 Impact of Replacing Duplicate and Wrong Images](#impact-of-replacing-duplicate-and-wrong-images)
#### [2.4 Train, Validation, and Test Split](#train-validation-and-test-split)
#### [2.5 Loading Data into Google Collaboratory](#loading-data-into-google-collaboratory)

### [3. Develop the Image Classification Model](#develop-the-image-classification-model)

#### [3.1 Baseline Model](#baseline-model)
#### [3.2 Baseline Model with Basic Data Augmentation](#baseline-model-with-basic-data-augmentation)
#### [3.3 Reducing Parameters for Model](#reducing-parameters-for-model)
#### [3.4 Tuning Batch Size for Model](#tuning-batch-size-for-model)
#### [3.5 Tuning Model Architecture](#tuning-model-architecture)
#### [3.6 Tuning Learning Rate](#tuning-learning-rate)
#### [3.7 Performing Advanced Augmentation Techniques](#performing-advanced-augmentation-techniques)
#### [3.8 Performing L1 & L2 Regularization](#performing-l1--l2-regularization)
#### [3.9 Adding Dropouts to Model](#adding-dropouts-to-model)
#### [3.10 Using Pre-trained Model without Data Augmentation](#using-pre-trained-model-without-data-augmentation)
#### [3.11 Using Pre-trained Model with Data Augmentation](#using-pre-trained-model-with-data-augmentation)
#### [3.12 Fine Tuning Pre-trained Model](#fine-tuning-pre-trained-model)

### [4. Evaluate models using Test Images](#evaluate-models-using-test-images)

### [5. Use the Best Model to perform Classification](#use-the-best-model-to-perform-classification)

### [6. Summary](#summary)

#### [6.1 Techniques that improved the model performance](#techniques-that-improved-the-model-performance)
#### [6.2 Future Improvements](#future-improvements)

# Overview

## Problem Statement

In this assignment, the problem statement revolves around creating an image classification model to recognize and categorize 10 different types of food items from a dataset of food images. Each student was assigned to different food items, and I was assigned the following items: beet_salad, beignets, ceviche, chocolate_mousse, cup_cakes, greek_salad, grilled_salmon, pancakes, panna_cotta, spaghetti_bolognese.

## Objective

The primary objective revolves around implementing various techniques learned in the lesson to build an effective image classification model. The goal extends beyond merely optimizing model performance metrics like accuracy and loss, it also involves emphasizing generalization and mitigating overfitting to ensure the model's robustness in classifying unseen data. Therefore, balancing optimization and generalization becomes crucial in this assignment. Optimization involves refining the model's performance metrics during training, while generalization aims to ensure the model can accurately classify new, unseen data by preventing overfitting to the training dataset's peculiarities.

To accomplish this objective, the assignment requires us to explore and experiment with different strategies that enhance model performance while documenting each step taken. Notably, the focus is not solely on achieving high accuracy but also on understanding which techniques contribute significantly to model improvement and which ones may not be as effective. The assignment encourages investigating methods, such as data augmentation, regularization techniques (like dropout or L1/L2 regularization), adjusting hyperparameters, exploring various network architectures, and leveraging transfer learning methods. Through systematic experimentation and documentation, I have to recognize the most effective approaches for enhancing model performance and generalization while understanding the limitations of certain techniques in achieving these goals.

## Approach

In approaching this assignment, I adopted a structured plan to methodically build and refine the food classification model. The process was divided into five distinct steps, each aimed at incrementally enhancing the model's performance while systematically exploring various techniques and strategies.

**Step 1:** Create a baseline model, setting the initial standard for performance metrics.

**Step 2:** Experiment with basic data augmentation techniques, exploring different model architectures, batch sizes, and learning rates. Additionally, I tested learning rate schedulers to optimize training efficiency.

**Step 3:** Implemented mixup augmentation and applied random erasing techniques on the training images to further enhance the model's ability to generalize and capture diverse features.

**Step 4:** Investigated different regularization techniques, including L1 regularizer, L2 regularizer, and dropout, to prevent overfitting and enhance the model's robustness.

**Step 5:** Used pre-trained models for feature extraction without augmentation, feature extraction with augmentation, and fine-tuning to leverage transfer learning to improve classification accuracy.

To ensure clarity and traceability, I created separate .ipynb files for each step in the Code Files folder. For example, for step 1, the file name is ASG1_Pranav_Vijitharan_1.ipynb and this is similar for the other steps as well.

For the code, I have used Google Collaboratory Pro to run my models since my laptop does not have GPU 1. Therefore, I made use of the T4 GPU in Google Collaboratory Pro. I purchased the Pro version as well so that the GPU doesn’t suddenly get disconnected. I would also like to thank Akul and Shawn for helping me run some of my models on their laptops as they have GPU 1 present on their laptop/PC.

Throughout this process, I tested numerous models and techniques, while documenting my findings, and the rationale behind each approach. This comprehensive report aims to cover my journey, highlighting the most effective techniques and insights gained while constructing the food classification model. By evaluating the impact of various methods, I aim to discover which strategies significantly contribute to model performance and understand their relevance in creating a robust and accurate classification model. Let's now delve into the report to find out more.

# Data Preprocessing and Data Loading

## Replace Duplicate Images

While I was creating some models, I explored the dataset by just scrolling through the images, and I realized that there were duplicate images present in the dataset. Duplicate images in the dataset can lead to biases in the model during training. If the same image appears multiple times, the model might excessively learn from these duplicate instances, prioritizing them over other crucial data. This could result in overfitting and reduce the model's ability to generalize well to unseen data. Therefore, I made a Python script to remove duplicate images. The filename of the script is Remove_Duplicates.py.

## Replace Wrong Images

While looking at the dataset, I also realized that there were images not related to the class. Inaccurate images can confuse the model during training. Removing such images ensures that the model learns from a clean and accurate dataset, leading to better classification performance.

**Asian Person in Cupcake class**

Before replacing the Asian person in the cupcake class, I decided to experiment. I ran the model with the Asian person still in the cupcake class and ran a model. Then, I tested the model with an online image of Jimmy O. Yang who is my favorite Asian standup comedian. The model predicted him as a cupcake which is quite hilarious.

After cleaning the dataset and replacing the Asian person from the dataset, I trained the same model once more without changing any hyperparameters and architecture and it still predicted Jimmy O. Yang as a cupcake, and I was surprised. One possibility is that there are some similarities between Jimmy O. Yang and cupcakes where both have some related features such as color etc.

**Noodle in Chocolate Mousse class**

## Impact of Replacing Duplicate and Wrong Images

To see the impact of replacing the images, I ran 2 models, 1 model is where the model trains with the raw images while another model trains with cleaned images.

Model trained on **raw images**:

Test Loss: 1.5089426040649414

Test Accuracy: 0.47600001096725464

Model trained on **cleaned images**:

Test Loss: 1.5203131437301636

Test Accuracy: 0.4779999852180481

There is not much of a great impact after cleaning the images. The removal of duplicate and incorrect images might not have substantially altered the dataset's overall quality. If the dataset was already sufficiently large, removing a small number of duplicates or incorrect images might not notably impact the model's performance.

## Train, Validation, and Test Split

I have used the Image_Preprocessing.ipynb file to extract the images that I was assigned, and it also helped me split the data into train, validation, and test sets. There are 7500 images in the train folder, 2000 images in the validation folder, and 500 images in the test folder.

## Loading Data into Google Collaboratory

To run models on Google Collaboratory, I first have to upload all the train, validation, and test folders that contain the images to my Google Drive. After that, I have to import the drive package from the google_colab library in the code so that I can mount my Google Drive on Colab and this will allow my code to access any files in my Google Drive.

# Develop the Image Classification Model

## Baseline Model

A baseline model establishes a starting point for model performance. It is the initial version of the model that is used to measure and compare the effectiveness of subsequent improvements or modifications. This model will help in setting realistic expectations for model enhancement.

Here are the hyperparameters that I have used for this baseline model:

-   Image Size - 224 (I will be using the same image size throughout)
-   Epochs - 30 (Smaller epoch as I know the model is going to overfit quickly)
-   Batch Size - 20
-   Learning Rate - 1e-4
-   Optimizer - RMSprop

I will also be using a callback function called Model Checkpoint which will allow me to save models based on the highest validation accuracy achieved during training and therefore allow me to capture the best-performing version of the model. I am using model checkpoint for all the models I create.

Here is the model architecture:

Total params: 9683658 (36.94 MB)

Trainable params: 9683658 (36.94 MB)

This baseline model started to overfit right away after the 5th epoch. Based on the training and validation loss, the model's training loss continues to decrease while the validation loss starts increasing after the 5th epoch. What this shows is that the model learns the train images way faster, and it learns to almost 100%. It learns all train images too well, but cannot generalize unseen data. The model checkpoint last saved the model at the 7th epoch where the validation accuracy was 0.518 and the validation loss was 1.4411.

I have used a somewhat complex model and there are around 9.6 million parameters in this model. Maybe this is a reason why the model overfitted very quickly due to the large network size. I will have to reduce the total number of parameters in my CNN and therefore reduce the size of the network later on to see if this delays overfitting.

## Baseline Model with Basic Data Augmentation

Data augmentation is a technique used to expand a dataset by applying various transformations or modifications to existing images. The goal of data augmentation is to increase the diversity of the dataset without collecting new data, thereby enhancing a model's ability to generalize and improve its performance. As more data is added by data augmentation, this allows the model to train with more images and therefore should delay overfitting. I have added Data Augmentation to the baseline model to check how effective this is. Below it shows the augmentation that I have done:

-   Rotation range = 40
-   Width shift range = 0.2
-   Height shift range = 0.2
-   Shear range = 0.2
-   Zoom range = 0.2
-   Horizontal flip = True

The baseline model with data augmentation was able to delay overfitting at around the 20th epoch while without augmentation overfitting occurred at the 5th epoch. As for the best validation accuracy, the model with Data Augmentation had an accuracy of 0.724, and without Data Augmentation had a validation accuracy of 0.518 and this shows a drastic improvement in accuracy after Data Augmentation. So, data augmentation seems to be effective.

## Reducing Parameters for Model

I have to find a way to reduce the parameters by simplifying the architecture of the model. By analyzing the baseline model architecture, I realized that the fully connected layer with 512 neurons contributes to the greatest number of parameters of close to 9.4 million. Therefore, I have removed that fully connected layer and built a model using only convolution layers. So, in this new architecture, the model will extract features using the convolution layers and after maxpooling and flattening, the features will be directly fed into the softmax activation function.

Here is the model architecture:

Total params: 425162 (1.62 MB)

Trainable params: 425162 (1.62 MB)

Non-trainable params: 0 (0.00 Byte)

There is a drastic change in the number of parameters as the baseline model had 9.6 million and the model without the fully connected layer has only 0.425 million.

I have also changed some hyperparameters compared to the previous model:

-   Epochs - 100 (Want to see how the model performs over a longer epoch)
-   Optimizer – Adam (Adam usually performs better in CNN

The model without the fully connected layer only started to overfit after 40 epochs which is better than the model with the fully connected layer as that model started to overfit after 20 epochs. Moreover, for the model without a fully connected layer, the validation loss began to increase but the gradient wasn’t that high compared to the other model where the gradient of increase in validation loss is higher.

As for the best validation accuracy, the model without a fully connected layer had an accuracy of 0.7612, and with a fully connected layer had a validation accuracy of 0.724. So, not using fully connected layers in the model seems to be effective. This performance improvement can be explained. The dense layer has introduced unnecessary complexity to the model and also might not have been well-suited for this dataset. Simplifying the model architecture by removing this layer might have made the model more suitable for the food classification task, leading to better generalization.

## Tuning Batch Size for Model

I have only tested out 2 batch sizes [25, 50]. Based on some research from the paper, “On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima”, the author claims that large batch methods tend to result in models that get stuck in local minima while smaller batches will more likely push out local minima and find the Global Minima, therefore, saying that small batch size is more effective (Nitish., 2017).

Based on the training and validation loss graphs, the model with a batch size of 50 starts to overfit after 50 epochs while the model with a batch size of 25 overfits after 40 epochs. However, based on the best validation accuracy, the model with batch size 50 has an accuracy of 0.7572, and the model with batch size 25 has an accuracy of 0.7612. This is a tough decision to make as both batch sizes have their pros and cons but later on, I chose to use batch size 25 as I decided to prioritize accuracy.

## Tuning Model Architecture

I have made 5 different model architectures, and these models have the same hyperparameters. Here are the hyperparameters that I have fixed for this experiment:

-   Image Size - 224
-   Batch Size - 25
-   Learning Rate – 0.0005
-   Optimizer – Adam

1st model architecture

Total params: 425162 (1.62 MB)
Trainable params: 425162 (1.62 MB)
Validation Loss: 0.9483
Validation Accuracy: 0.7612

This 1st model architecture is very basic and yet the validation accuracy is quite high where the best validation accuracy is 0.7612. For this model architecture, overfitting occurs at around the 40th epoch. This basic and simple architecture can perform well in terms of accuracy, but the loss can still be reduced further to further improve the performance of the model.

2nd model architecture

Total params: 1024298 (3.91 MB)
Trainable params: 1024298 (3.91 MB)
Validation Loss: 0.7620
Validation Accuracy: 0.7712

The 2nd model architecture is quite like the 1st model architecture, but the depth of the convolution layers is different as the 2nd model architecture has more 1 more convolution layer than the 1st model architecture. Therefore, the 2nd model architecture can be considered as a deeper network with more layers which increases the complexity due to its repeated convolution layers. The 2nd model architecture also has a greater number of parameters 2.5x more than the 1st model architecture. The complexity of this model was helpful as it allowed the model to perform better than the 1st model architecture as it made the validation loss decrease from 0.9483 to 0.7620. Therefore, the 2nd model architecture is outstanding.

3rd model architecture

Total params: 1428266 (5.45 MB)
Trainable params: 1428266 (5.45 MB)
Validation Loss: 0.9221
Validation Accuracy: 0.7357

The 3rd model architecture has 2 additional convolution layers of 256 filters and corresponding MaxPooling2D layers compared to the 2nd model architecture. The thought process when creating this model is to increase the complexity of the model so that it can capture more intricate patterns in the images compared to the 2nd model architecture. However, after running the model, it shows that the 3rd model architecture resulted in a decrease in the model performance compared to the 2nd model architecture. Based on the chart, the model started to overfit slightly only after the 50th epoch and this is better compared to the 2nd model architecture as that model overfitted faster at around the 40th epoch. But the 3rd model architecture had a higher validation loss and lower validation accuracy than the 2nd model architecture which shows that the 3rd model architecture is not great. This experiment shows that sometimes, a complex architecture might not be suitable for the given problem. A simpler architecture might be more effective in capturing the relevant features or patterns within the dataset.

4th model architecture

Total params: 2434346 (9.29 MB)
Trainable params: 2434346 (9.29 MB)
Validation Loss: 1.023
Validation Accuracy: 0.7482

The 4th model architecture has 1 additional convolution layer of 512 filters and corresponding MaxPooling2D layers compared to the 3rd model architecture. The purpose of increasing the complexity of the model is just to document what are the results. Compared to the 3rd model architecture, the 4th model architecture seemed to improve the validation accuracy, but this model still didn’t achieve a better result than the 2nd model architecture. However, in terms of validation loss, this model performed poorly than both the 2nd and 3rd model architecture. Therefore, I would not be making any more complex models based on the dataset and features extracted, a model architecture with smaller parameters and a simpler network tends to perform better.

5th model architecture

Total params: 1863338 (7.11 MB)
Trainable params: 1863338 (7.11 MB)
Validation Loss: 0.9144
Validation Accuracy: 0.7045

\*I could only run the model till the 50th epoch as my Google Collab Compute Units were going low

The 5th model architecture has fewer convolution layers compared to the 2nd model architecture and yet it has larger parameters than the 2nd model. This experiment made me realize that the total number of parameters in a neural network is not solely determined by the number of layers but it's influenced by various factors like layer configurations, input shapes, etc. As a result, a seemingly simpler architecture of model 5 might have more parameters than expected due to specific layer settings or transformations within the network. Now to compare the output of the model, it will not be fair to compare this model to the 2nd model architecture as that was run till the 100th epoch. Therefore, to be fair, I will compare the validation accuracy and loss to the 2nd model architecture when it was in the 50th epoch.For the 2nd model architecture when it was at the 50th epoch, it had a validation accuracy of 0.7335 and validation loss of 0.8248 and this is better than the 5th model architecture’s validation accuracy and loss at the 5th epoch.

Conclusion

Based on all the experiments, model 2 architecture proved to work the best for this context as it is neither too simple nor too complex and it is the perfect model architecture that can have a low validation loss and high validation accuracy.

## Tuning Learning Rate

Once I finalize my model architecture, it is time to tune the learning rate. In this experiment, I have tested out the following learning rates, 0.0005, 0.001, and 1e-5. I have also experimented with learning rate schedulers, specifically step decay.



Validation Loss: 0.7620
Validation Accuracy: 0.7712


Validation Loss: 1.1648
Validation Accuracy: 0.7015

Validation Loss: 1.5841
Validation Accuracy: 0.4630

Based on all the charts and metrics, it looks like a learning rate of 0.0005 is the most appropriate compared to the other learning rates.

I have also done some exploration and learned about learning rate schedulers and learning rate decay. What learning rate decay does is that it trains the network with a large learning rate and then slowly reduces it until local minima is obtained.

![image](https://github.com/Pranav-Vijitharan/Food-Classification-Model/assets/122760008/e6827464-44a8-483f-a31a-89870742e851)

The image above (Vaibhav,. 2020) shows two lines. The blue line uses a constant learning rate and usually, the steps taken by our algorithm while iterating towards minima are so noisy that after certain iterations it seems to wander around the minima and does not converge. The green line uses a learning decay algorithm and what it shows is that since the learning rate is large initially it has relatively fast learning but as it tends towards minima learning rate gets smaller and smaller and ends up oscillating in a tighter region around minima rather than wandering far away from it.

In this experiment, I used a learning rate decay algorithm called step decay. Step decay systematically reduces the learning rate at specific intervals or "steps" during the training process. The steps can be defined in the code. Below, it shows how I implemented step decay.

![A computer screen with white text Description automatically generated](media/e415117a6f74ac7d97008853a3e93fe1.png)

I stated that the initial learning rate should be 0.0005 and the learning rate should decrease by 50% after every 10 epochs. The chart below shows how the learning rate changed when training the model.

![A graph with a line Description automatically generated](media/63ff54aee0559e6e4c78492f8e42c111.png)

![A graph of training and validation Description automatically generated](media/318a118a784cbb936f4e614cd0927e4e.png)![A graph of a training and validation accuracy Description automatically generated](media/22359f00a54aeacc823f31ed0143eb0d.png)

Validation Loss: 0.8547

Validation Accuracy: 0.7285

Based on the graph, after the 40th epoch, the validation loss became very stagnant and similar to the validation accuracy. This is because the learning rate was so small that it became close to negligible, therefore, this does not allow the weights to get updated therefore the loss and accuracy become constant. To avoid this, I should have used a smaller drop like 0.2 and also increased the initial learning rate to 0.01 and this would have been more effective.

## Performing Advanced Augmentation Techniques

There are various data augmentation techniques and the ones that I will be focusing on are Mixup augmentation and Random Eraser. The reason for trying out these techniques is to try to regularize the models further and also improve the generalization of the model so that it improves the model’s robustness and performs well on unseen images as well.

Mixup Augmentation

Mixup generates new training samples by combining pairs of images.

![A screenshot of a computer Description automatically generated](media/7b22967a2d8023a9b1ef3bdeeedae438.png)

The above image shows the output of images once mixup augmentation has been done. For example, Image 1 is a combination of the 6th and 8th class and the proportion of class 6 is 64% while class 8 is 35%. The proportion of classes in each image can be different as it is randomized. In order to accomplish this, I had to create a class called MixupImageDataGenerator (refer to ASG1_Pranav_Vijitharan_3.ipynb) In the class there are 2 methods called mixup_augmentation and flow_from_directory. The mixup_augmentation method will take care of the mixing of images.

How mixup works in detail:

1.  First random coefficients λ will be generated from a beta distribution. These coefficients determine the blending ratio between two images.
2.  The new mixed image is generated pixel-wise by interpolating between the pixel values of the two original images. For each pixel location, the value is computed as λ \* image1 + (1 - λ) \* image2.
3.  Corresponding labels are also mixed similarly to the images. For classification tasks, where labels are one-hot encoded, Mixup blends the labels in the same proportion as the images.

![A graph of training and validation Description automatically generated](media/d90bd7ec5e40e1e970abb57d2e2a1612.png)![A graph of a training and validation accuracy Description automatically generated](media/897a87b721e739e955f948de7a10dcd8.png)

Validation Loss: 0.7920

Validation Accuracy: 0.7400

Doing mixup augmentation did wonders for the model. It made the validation loss lower than the training loss and in fact, letting the model run for more epochs would have given a better result. Moreover, the training accuracy and validation accuracy converged showing that the model has reached a stable state, performing well on both seen and unseen data. I have also noticed that the training accuracy is lower than the validation accuracy until the end when they finally converged. During the early training epochs, mixup may introduce some instability as the model adapts to the augmented samples. This instability might lead to lower performance in the training accuracy and loss. However, in the long run, when the model converges, mixup often leads to better generalization, lower overfitting, and improved performance on unseen data.

Random Eraser Augmentation

Random Erasing technique is used to randomly select and erase patches of an image within the training dataset, forcing the model to learn features apart from the missing areas.

![A collage of food Description automatically generated](media/7125a0ee803661400e1179773d6ee5ec.png)

The image above shows the output of images once random eraser augmentation has been done. There are random patches of boxes covering the image and this is the random eraser augmentation. To do random eraser augmentation, I created a function get_random_eraser (refer to ASG1_Pranav_Vijitharan_3.ipynb).

The parameter for the function is p = 0.5, s_l = 0.02, s_h = 0.4, r_1 = 0.3, r_2 = 1/0.3, v_l = 0, v_h = 255.

-   p: Probability (p) of applying the random erasing. If a randomly generated probability is higher than p, the function returns the original image without applying any erasing.
-   s_l and s_h: Lower and higher limits for the proportion of the erased area relative to the total image area. r_1 and r_2: Lower and higher bounds of the aspect ratio range for the erased area.
-   v_l and v_h: Lower and higher bounds for the pixel value used to fill the erased area.

There is another inner function called eraser which takes an input image as an argument and performs the erasing operation based on the defined parameters. It generates random values for the size, aspect ratio, position, and pixel value to erase a patch from the input image.

For the parameters like p, s_h, and r_1, I used 0.5, 0.4, and 0.3 respectively as based on a research paper, using these numbers result in the lowest test errors. The charts below show a bar chart where for each parameter, the researchers tried using different values and calculated the test errors (Zhong, 2020).

![A graph with blue and green bars Description automatically generated](media/5aea0993557b5846b23fc549f735a39b.png)

![A graph of a training and validation accuracy Description automatically generated](media/92d4db03822e5c085bf0f8ee53f2c382.png)

![A graph of training and validation Description automatically generated](media/b6a8eec91768a4b7b6ed9dd8fa3dce77.png)

Validation Loss: 0.8758

Validation Accuracy: 0.7315

Random Eraser has made some improvements to the model as it managed to reduce the validation loss compared to the model that did not do Remove Eraser Augmentation.

Conclusion

Mixup augmentation is more effective than Random Eraser due to several reasons. Mixup augmentation proves that is a very powerful technique to regularize the model and makes the model generalize well on data that it has not seen. Random eraser is effective in introducing robustness to the model but might not inherently encourage the model to learn more generalizable features as significantly as mixup does. Therefore, mixup is a very powerful technique to use when creating a model

## Performing L1 & L2 Regularization

L1 (Lasso) and L2 (Ridge) regularization methods add penalty terms to the loss function based on the magnitudes of model weights. They discourage overly large weights, promoting sparsity (L1) or smoothness (L2) in the learned parameters.

L1 regularization tends to make some weights zero and this makes it have a "shrinkage" effect, leading to feature selection by driving less important or redundant features' weights to zero. This results in a more interpretable and potentially simpler model.

L2 regularization makes all weights smaller but does not drive them to 0. It penalizes large weights but does not favor sparsity as aggressively as L1 regularization. L2 generally results in smoother weight distributions and better optimization.

![A graph of a graph with blue dots Description automatically generated with medium confidence](media/345a1b149fabb8c23d9a3d3c3e5522b8.png)![A computer code with many small colored text Description automatically generated with medium confidence](media/331bc4f6bac06464954c17658b34ffbc.png)L1 Regularisation

![A graph of training and validation Description automatically generated](media/81242592fa6b07da2f4ed6a279272f55.png)

Validation Loss: 1.9571

Validation Accuracy: 0.3360

Using L1 regularisation is not helpful as the training and validation loss is still very high and the loss does not seem to decrease. This can occur because the regulariser penalizes the weights and makes them 0 for the nodes and this will simplify the model too much therefore, the convolution layers will not be able to identify the features in the images and that will impact the CNN model very negatively.

![A graph of training and validation Description automatically generated](media/043f31144aa5a26848fcced8cb273ff6.png)![A computer code with many colorful text Description automatically generated with medium confidence](media/cd95808973d149116891ad927770bd02.png)L2 Regularisation

![A graph of a training and validation accuracy Description automatically generated](media/dc6ec305feae4161bd83b0b30e4e0cb6.png)

Validation Loss: 1.0812

Validation Accuracy: 0.6795

L2 regularisation performs well in this scenario as by referring to the graph, both the training and validation accuracy chart and the training and validation loss chart, the lines are closely aligned. This indicates that the model is well-generalized, neither underfitting nor overfitting. In fact, I can further increase the epoch in this model as it can further improve.

Conclusion

Therefore, in regularisers, L2 proved to be more useful than L1 regularisers based on the model performance. L2 was able to perform well as it makes all weights smaller but does not drive them to 0. Therefore, this characteristic in L2 allows more information to be preserved which can contribute to better retention of useful features and representations in the CNN model.

## Adding Dropouts to Model

![A screen shot of a computer program Description automatically generated](media/0d8c9f42fc33d61d3eecbec10a12067c.png)Dropout is a regularization technique used in neural networks to prevent overfitting. It works by randomly deactivating or "dropping out" a fraction of neurons (along with their connections) in a layer during each training iteration. This technique introduces randomness into the network, forcing it to learn a more robust and generalized set of features.

![A graph of training and validation loss Description automatically generated](media/efca003437c442ef28d08721c747030f.png)

![A graph of a training and validation accuracy Description automatically generated](media/ef752ce235888093e941aaacb26d0617.png)

Validation Loss: 1.1879

Validation Accuracy: 0.6095

Random dropout helps prevent overfitting by making the network less sensitive to the precise details of any given training example. Since my model only has convolution layers and maxpooling layers, adding dropouts does not necessarily help the model. Additionally, I do not have fully connected layers in my model architecture and dropouts will be more helpful if applied to the fully connected layers. Therefore, having dropouts in my model architecture will not be well suited.

## Using Pre-trained Model without Data Augmentation

\*I could only run the model till the 30th epoch from now on as my Google Collab Compute Units were going low

I have used the VGG16 pre-trained model to extract the features from the images. These extracted features will then be directly passed to the model which will then classify the images.

Here are the hyperparameters that I have used for this baseline model:

-   Image Size - 224
-   Epochs - 30
-   Batch Size - 20
-   Learning Rate - 2e-5
-   Optimizer - Adam

![A screen shot of a computer code Description automatically generated](media/85374a55742c8b7d8d47611b4f61bc65.png)

Total params: 6425354 (24.51 MB)

Trainable params: 6425354 (24.51 MB)

Non-trainable params: 0 (0.00 Byte)

![A graph of training and validation Description automatically generated](media/857e63c1ef893b360e8c3584b7a9ff6c.png)

![A graph of a line Description automatically generated](media/203538347187f42da9f54fba283712e5.png)

Validation Loss: 0.9579

Validation Accuracy: 0.6890

This model reached a validation accuracy of about 68%, which is quite average as I have made models from scratch that are better than this pre-train model’s accuracy. Additionally, the plots also indicate that overfitting occurs almost from the start, despite using dropout with a large rate. This is because this technique does not leverage data augmentation, which is essential to preventing overfitting.

## Using Pre-trained Model with Data Augmentation

I have added the VGG16 model to the model architecture. However, since the parameters for the pre-train model are very high, I freeze all the layers in the pre-train model. If I didn't do this, then the representations that were previously learned by the convolutional base would get modified during training and that would be no point in using the pre-train model. This model will now allow me to do data augmentation as well. Below it shows the augmentation that I have done to the train dataset:

-   Rotation range = 40
-   Width shift range = 0.2
-   Height shift range = 0.2
-   Shear range = 0.2
-   Zoom range = 0.2
-   Horizontal flip = True

Model 1

![A screen shot of a computer program Description automatically generated](media/c66a893d3fc2847166a8fcc0e6b25867.png)

Total params: 21140042 (80.64 MB)

Trainable params: 6425354 (24.51 MB)

![A graph of training and validation Description automatically generated](media/1ad7da125f1bfffb99e7412aecdc5f32.png)![A graph with blue dots Description automatically generated](media/decb044190ed558e4fd6c3163e9dd9c2.png)Non-trainable params: 14714688 (56.13 MB)

Validation Loss: 0.8331

Validation Accuracy: 0.7090

Data Augmentation has improved the model by preventing overfitting of the model as well as decreasing the validation loss tremendously from 0.9579 to 0.83331. However, there was only a minor improvement in the validation accuracy. Therefore, I will explore how to increase the accuracy.

Model 2

![A screen shot of a computer program Description automatically generated](media/2d4d6a518351dcc08aed8963124b863d.png)

Total params: 14881738 (56.77 MB)

Trainable params: 166282 (649.54 KB)

![A graph of training and validation Description automatically generated](media/3da62c0952543e9d8573ff3fe6134fb0.png)Non-trainable params: 14715456 (56.14 MB)

![A graph of training and validation accuracy Description automatically generated](media/6eebb5e0c00d4d26e02179d5097f45ce.png)

Validation Loss: 1.2884

Validation Accuracy: 0.5560

In this model, I have added 2 new layers called GlobalAveragePooling2D and BatchNormalisation. GlobalAveragePooling2D condenses the spatial dimensions of each feature map created by the pre-train model to a single value, simplifying the representation of learned features before feeding them into subsequent layers. Batch Normalization is a technique used during training to normalize the activations within each layer of a neural network. I have also created 2 dense layers, one has 256 nodes while the next dense layer has 128 nodes. Despite having more dense layers, the parameters are much lower, and this is because of the GlobalAveragePooling2D layer which helped reduce the spatial dimension. By changing the architecture, I was able to make the validation loss lower than the training loss and the validation accuracy higher than the training accuracy. Letting the model run for more epochs would have given a better result too.

Model 3

![](media/e544bb627c60a49afea55d177e81af10.png)

![A screen shot of a computer program Description automatically generated](media/72c8fc9050c93b9fd11a29b5e7386f83.png)

Total params: 2621642 (10.00 MB)

Trainable params: 362890 (1.38 MB)

![A graph of training and validation Description automatically generated](media/b5812908ed4429a14370ec77379f9d59.png)Non-trainable params: 2258752 (8.62 MB)

![A graph with blue dots Description automatically generated](media/e34e0b7b09b28ae5a0530b7285fe7d93.png)

Validation Loss: 0.5499

Validation Accuracy: 0.8120

Model 3 compared to model 2, I changed the pre-train model from VGG16 to mobilenetv2_1.00_224. This change led to breaking records as officially, this pretrained model’s validation accuracy and loss have overtaken my best scratch model’s validation accuracy and loss. Not only that, but the model also reached this accuracy in just 30 epochs as well.

## Fine Tuning Pre-trained Model

Model 1

![A screen shot of a computer screen Description automatically generated](media/22bde0921c5433dd2a8cbd2f534da0ce.png)

Total params: 2257984 (8.61 MB)

Trainable params: 2223872 (8.48 MB)

![A graph of training and validation Description automatically generated](media/32442814b6578b6d2885ffef28e88694.png)Non-trainable params: 34112 (133.25 KB)

![A graph of a graph with blue dots Description automatically generated](media/532cc879475e6867a1af9e24a9d77893.png)

Validation Loss: 0.5511

Validation Accuracy: 0.8335

I have only unfrozen the last 3 layers and yet the trainable parameters are 2.22 million while the non-trainable parameters are 34, 000.

Model 2

![A screen shot of a computer program Description automatically generated](media/dabdce469b75e4e8caf1874792977d6e.png)

Total params: 2621642 (10.00 MB)

Trainable params: 775690 (2.96 MB)

Non-trainable params: 1845952 (7.04 MB)

![A line graph with blue dots Description automatically generated](media/cc33b51b58b131528f1c65a8dccf8379.png)![A graph with blue dots Description automatically generated](media/e1207e567d25623b7bb75c225642d0c2.png)

Validation Loss: 0.6473

Validation Accuracy: 0.8535

Now, I have unfrozen the last 4 layers and the trainable parameters are 775,000 while the non-trainable parameters are 1.8 million. Doing this increased the validation accuracy compared to model 1 but also increased the validation loss as well.

# Evaluate models using Test Images

**Baseline Model**

(ASG1_Pranav_Vijitharan_1.ipynb)

**Test Loss: 1.5203131437301636**

**Test Accuracy: 0.4779999852180481**

\* F1-score is the harmonic mean of precision and recall,

providing a balance between the two metrics.

\* Overall accuracy of the model in correctly predicting all classes.

![A screenshot of a computer screen Description automatically generated](media/839da063cb42e1f9b37702e08d063e83.png)

-   Spaghetti Bolognese has the highest F1 score of 0.74.
    -   Chocolate Mousse, Ceviche, Panna Cotta and Cup Cakes have a quite low F1 score as they are in the 30 range.
        -   The overall accuracy is 0.47 which is low and will have to be improved.

\* For confusion matrix the diagonal should have higher numbers

\* The rest of the numbers should be close to 0

![A colorful squares with numbers Description automatically generated](media/a1817f1c9846a8b71318776b37372d35.png)

-   The confusion matrix shows that there are many numbers around apart from the diagonal and this shows that there is a lot of Type I and Type II error present.
    -   On the diagonal no square has the number 40 and above.
        -   There is not a proper diagonal that is formed.
            -   There is still room for improvement.

**Model with Good Architecture and Hyperparameter**

(ASG1_Pranav_Vijitharan_2.ipynb)

Trial 4

**Test Loss: 0.7998452186584473**

**Test Accuracy: 0.7519999742507935**

![](media/2b79d220f21bad4ac4d394cc9487cd0d.png)

-   Spaghetti Bolognese has the highest F1 score of 0.92.
    -   Chocolate Mouse has a low F1 score compared to other classes.
        -   The overall accuracy is 0.76 which is good but can still be improved.

![A graph of numbers and labels Description automatically generated](media/9a03784c95f7172b9c44c3ac9a13125d.png)

-   On the diagonal 5 squares has the number 40 and above.
    -   A diagonal line can be seen more clearly.
        -   Class 3 often gets confused with Class 4 and 8.
            -   Class 2 gets confused with class 5.
                -   Class 8 gets confused with class 3

**Model with Mixup Augmentation**

(ASG1_Pranav_Vijitharan_3.ipynb)

**Test Loss: 0.8052988648414612**

**Test Accuracy: 0.7400000095367432**

![](media/c33bbc88711a1fa61411402554e19a84.png)

-   Spaghetti Bolognese has the highest F1 score of 0.93.
    -   Chocolate Mouse and Panna Cotta has a low F1 score compared to other classes.
        -   Chocolate Mousse has improved compared to before, but Panna Cotta remain the same.
            -   The overall accuracy is 0.75 which has reduced then before.

![A graph of a number of labels Description automatically generated with medium confidence](media/7161563ad8f36b7304530c094c624be8.png)

-   On the diagonal, 4 squares has the number 40 and above and this is a drop than before.
    -   A diagonal line can still be seen more clearly.
        -   Class 3 often gets confused with Class 4 and 8.
            -   Class 2 gets confused with Class 5
                -   Class 8 gets confused with class 3.

**Pre-trained Model that is finetuned**

(ASG1_Pranav_Vijitharan_5.ipynb)

model_8

**Test Loss: 0.6233285069465637**

**Test Accuracy: 0.8320000171661377**

![](media/98e29c2db56ec836f747a519229ecc58.png)

-   Spaghetti Bolognese has the highest F1 score of 0.97.
    -   Chocolate Mouse and Panna Cotta has a low F1 score compared to other classes.
        -   The overall accuracy is 0.83.

![A graph of numbers and labels Description automatically generated](media/03a2aa7088e97b333c86dc87101032bc.png)

-   On the diagonal, 8 squares have the number 40 and above.
    -   A diagonal line can be seen very clearly.
        -   Class 3 often gets confused with Class 8.
            -   Class 5 gets confused with Class 2
                -   Class 8 gets confused with class 3.

**Conclusion**

Based on the test evaluation, the pretrained model that is finetuned is the best model as it not only has a highest test accuracy compared to other models but also has the lowest test loss.

# Use the Best Model to perform Classification

![A collage of images of food Description automatically generated](media/6be48d0e1a1a21a4f5749633c5441f1c.png)

Based on these images, out of the 15 images I took from online, 2 of them were predicted wrongly by the model. The ones that are wrong are greek salad1 and pancake1. The model predicted pancakes as chocolate mousse and greek salad as ceviche.

I was able to understand why the model predicted the greek salad wrongly as the picture I provided was tough with a person being in the background. Not only that, greek salad and ceviche somewhat look similar as well and therefore I was able to understand why the model predicted that wrongly.

However, what I am not able to comprehend is the other image that the model predicted wrongly which is the pancakes. I gave the model of a pancake with berries on top of it and there were no disturbances or obstruction to the photo. But it predicted the pancake as chocolate mousse. This did not make much sense to me, and I was baffled. Moreover, the other pancake image that I gave the model was much harder as it had Dwayne Johnson eating the pancake and he was a big obstruction in the photo. However, the model was still able to predict the image of Dwayne Johnson eating a pancake as a pancake. Then coming back to the pancake image that was predicted wrongly the pancake did not seem to resemble a chocolate mousse as well.

But overall, the model was able to predict most of the images correctly and predicted a lot of tough images correctly as well and I am satisfied with this output.

# Summary

## Techniques that improved the model performance

1.  Performing Basic Data Augmentation together with Mixup.
2.  Using learning rate of 0.0005.
3.  Using a batch size of 25.
4.  Not using a fully connected layer in the model.
5.  Using mobilenetv2_1.00_224 for pretrain model.
6.  When fine-tuning the pretrain model, unfreezing the last 4 layers helped.

## Future Improvements

1.  Using Mixup augmentation when using pretrained model.
2.  Using Keras Tuner to tune hyperparameters of deep learning models.
3.  Trying out Batch Normalisation and GlobalAvgPool layers in models that had to be created from scratch.
4.  Focusing more on Pretrain models and finding suitable hyperparameters for the pretrain models.
