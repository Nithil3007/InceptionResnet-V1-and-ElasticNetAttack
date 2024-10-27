---------------------------------------------------------------

# [Model](https://iq.opengenus.org/inception-resnet-v1/)

The Inception-ResNet like the Inception v4 uses the same two reduction blocks. But uses inception-resnet blocks instead. This is done so that the output of the inception module is added to the input. Each of these inception-resnet blocks consist of a shortcut connection. This greatly reduces the effect of vanishing gradient problem, i.e. helps the model to memorize the patterns easier no matter how deep the neural network is. The network becomes selective of the patterns it wants to learn. Reduction blocks control the breadth and depth of the network by providing max pooling and convolutional filters.

Reduction block A has
1. a 3x3 MaxPool filter,
2. a 3x3 convolutional filter,
3. a 1x1 convolutional filter followed by 2 3x3 convolutional filters.

Reduction block B has
1. a 3x3 MaxPool filter
2. a 1x1 convolutional layer followed by a 3x3 convolutional layer.
3. a 1x1 convolutional layer followed by a 3x3 convolutional layer with different number of channels in output.
4. a 1x1 convolutional layer followed by 2 3x3 convolutional layers one over another.

![image](https://user-images.githubusercontent.com/102175607/202896131-4ed5f1a7-3899-414c-a488-0b97e35aabb4.png)

Figure 1. Inception-ResNet modules: A, B, and C 

![image](https://user-images.githubusercontent.com/102175607/202896153-aa6886cb-908e-4496-8aaf-27616e895e5c.png)

Figure 2. Reduction Blocks A and B

![image](https://user-images.githubusercontent.com/102175607/202896166-1fa8294a-b0c3-4bc6-9ccb-a83228f9d100.png)

Figure 3. Pre-processing Unit (Stem)

![image](https://user-images.githubusercontent.com/102175607/202896184-257c1ce5-7d00-494e-92c5-57cf59496cf0.png)

Figure 4. Overall model

---------------------------------------------------------------

# ElasticNet Attack

Adversarial attacks deceive the model into giving away sensitive information, making incorrect predictions, or corrupting them. Decisions taken by networks in classification can be manipulated by adding carefully crafted noise to an image which we often refer to as an ‘adversarial attack’ on a neural network. If done well this noise is barely perceptible and can fool the classifier into looking at a certain object and thinking that it is a totally different object. We have used untargeted attacks to corrupt the images, i.e., the goal is simply to make the target model misclassify by predicting the adversarial example as a class other than the original class.

Experimental results on MNIST, CIFAR-10, and ImageNet show that Elastic-net Attack to Deep neural networks (EAD) yields a distinct set of adversarial examples. More importantly, EAD leads to improved attack transferability suggesting novel insights on leveraging L1 distortion in generating robust adversarial examples.

ElasticNet Attack - https://arxiv.org/abs/1709.04114

---------------------------------------------------------------

### [Code](https://github.com/Nithil3007/InceptionResnet-V1-and-ElasticNetAttack)

---------------------------------------------------------------

# Dataset
[Retinal OCT Images (optical coherence tomography)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

--------------------------------------------------------------- 

# Files

[InceptionResnetV1 Pytorch (Config 1).ipynb - All the layers in the model are trainable.](https://github.com/Nithil3007/InceptionResnet-V1-and-ElasticNetAttack/blob/main/InceptionResnetV1%20Pytorch%20(Config%201).ipynb)

[InceptionResnetV1 Pytorch (Config 2).ipynb - The first 9 layers in the model are freezed.](https://github.com/Nithil3007/InceptionResnet-V1-and-ElasticNetAttack/blob/main/InceptionResnetV1%20Pytorch%20(Config%202).ipynb)

[Elastic net attack - without training.ipynb - The model is not trained to adapt elastic net attack. Poor results are obtained.](https://github.com/Nithil3007/InceptionResnet-V1-and-ElasticNetAttack/blob/main/Elastic%20net%20attack%20without%20training.ipynb)

[Elastic net attack - with training.ipynb - The model is trained to adapt the ElasticNet attack. Imporved results are obtained.](https://github.com/Nithil3007/InceptionResnet-V1-and-ElasticNetAttack/blob/main/Elastic%20net%20attack%20with%20training.ipynb)

---------------------------------------------------------------

# Project by

  1. Nithil Eshwar Mani
  2. Chiranjeevi
  3. Abiramashree

From Department of Computer Science & Engineering, College of Engineering, Guindy, Anna University
