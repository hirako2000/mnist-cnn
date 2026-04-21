## LeNet-5 → ConvNeXT

| # | Technique / Architecture | Year | Benefit | Paper | Category | Milestone | Dataset |
|---|--------------------------|------|---------|-------|----------|-----------|---------|
| 1 | LeNet-5 (Baseline) | 1998 | Original convolutional architecture | [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (LeCun et al., 1998) | Architecture | **LeNet-5** | MNIST |
| 2 | ReLU Activation | 2010 | Solves vanishing gradient problem | [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) (Nair & Hinton, ICML 2010) | Activation | LeNet-5 | MNIST |
| 3 | Xavier Init | 2010 | Stable gradient variance | [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (Glorot & Bengio, AISTATS 2010) | Initialization | LeNet-5 | MNIST |
| 4 | Dropout | 2012 | Prevents overfitting | Hinton et al., "Improving neural networks by preventing co-adaptation of feature detectors" (arXiv 2012) | Regularization | LeNet-5 | MNIST |
| 5 | Max Pooling | 1980s/1998 | Preserves sharp features; popularized by AlexNet | Fukushima, "Neocognitron" (1980); LeCun et al., LeNet-5 (1998); Krizhevsky et al., AlexNet (2012) | Pooling | LeNet-5 | MNIST |
| 6 | Data Augmentation | 2012 | Better generalization | Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, NIPS 2012) | Data Processing | LeNet-5 | MNIST |
| 6.1 | MixUp / CutMix | 2018/2019 | Advanced data augmentation (mix samples) | Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018); Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (ICCV 2019) | Data Processing | LeNet-5 | MNIST |
| 7 | Local Response Normalization (LRN) | 2012 | Channel-wise normalization (deprecated after BN) | Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, NIPS 2012) | Normalization | LeNet-5 | MNIST |
| 8 | AlexNet Architecture | 2012 | Deeper, wider, 2-group conv for GPU memory | Krizhevsky et al., NIPS 2012 | Architecture | **AlexNet** | **Tiny ImageNet** |
| 9 | 3×3 Conv Stacking (VGG-style) | 2014 | Increased depth with small kernels | Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (ICLR 2015) | Architecture | AlexNet | Tiny ImageNet |
| 9.1 | Inception (1×1 Conv) | 2014 | Bottleneck for channel reduction, reduces parameters | Szegedy et al., "Going Deeper with Convolutions" (GoogLeNet, CVPR 2015) | Architecture | AlexNet | Tiny ImageNet |
| 10 | Gradient Clipping | 2013 | Prevents exploding gradients (primarily for RNNs) | Pascanu et al., "On the difficulty of training recurrent neural networks" (ICML 2013) | Training Strategy | AlexNet | Tiny ImageNet |
| 11 | Adam Optimizer | 2014 | Adaptive learning rates | Kingma & Ba, "Adam: A Method for Stochastic Optimization" (ICLR 2015) | Optimization | AlexNet | Tiny ImageNet |
| 11.1 | AdamW (Weight Decay) | 2017 | Decoupled weight decay for better generalization | Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2017) | Optimization | AlexNet | Tiny ImageNet |
| 12 | Learning Rate Scheduling | 2015 | Systematic LR reduction | Standard practice; popularized by AlexNet (2012) | Optimization | AlexNet | Tiny ImageNet |
| 13 | Global Average Pooling | 2013 | Replace FC layers | Lin et al., "Network In Network" (ICLR 2014, preprint 2013) | Pooling | AlexNet | Tiny ImageNet |
| 14 | Batch Normalization | 2015 | Faster training, higher LR | Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (ICML 2015) | Normalization | AlexNet | Tiny ImageNet |
| 15 | He Init | 2015 | ReLU-specific initialization | He et al., "Delving Deep into Rectifiers" (ICCV 2015) | Initialization | AlexNet | Tiny ImageNet |
| 16 | ResNet Architecture | 2015 | Enables very deep networks (152 layers) | He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016) | Architecture | **ResNet** | Tiny ImageNet |
| 17 | Residual Connections | 2015 | Enables very deep networks | He et al., "Deep Residual Learning for Image Recognition" (ResNet, CVPR 2016) | Architecture | ResNet | Tiny ImageNet |
| 18 | Stochastic Depth | 2016 | Dropout for residual blocks | Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016) | Regularization | ResNet | Tiny ImageNet |
| 19 | Label Smoothing | 2016 | Prevents overconfidence | Szegedy et al., "Rethinking the Inception Architecture" (Inception-v3, CVPR 2016) | Regularization | ResNet | Tiny ImageNet |
| 20 | GELU Activation | 2016 | Smoother than ReLU | Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (arXiv 2016) | Activation | ResNet | Tiny ImageNet |
| 21 | Layer Normalization | 2016 | Better for smaller batches; adopted for CNNs in ConvNeXt | Ba et al., "Layer Normalization" (arXiv 2016) | Normalization | ResNet | Tiny ImageNet |
| 22 | Depthwise Convolution | 2017 | Separates spatial/channel mixing | Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions" (CVPR 2017) | Architecture | ResNet | Tiny ImageNet |
| 22.1 | Grouped Convolutions (ResNeXt) | 2017 | Cardinality dimension for better representation | Xie et al., "Aggregated Residual Transformations for Deep Neural Networks" (ResNeXt, CVPR 2017) | Architecture | ResNet | Tiny ImageNet |
| 23 | Cosine Annealing | 2017 | Better LR schedule with warm restarts | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017) | Optimization | **MobileNetV2-style** | Tiny ImageNet |
| 24 | Inverted Bottleneck | 2018 | Expand → Conv → Project | Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018) | Architecture | MobileNetV2 | Tiny ImageNet |
| 25 | Squeeze-and-Excitation (SE) | 2018 | Channel attention | Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018) | Architecture | MobileNetV2 | Tiny ImageNet |
| 26 | Stochastic Weight Averaging (SWA) | 2018 | Flatter minima, better generalization | Izmailov et al., "Averaging Weights Leads to Wider Optima" (UAI 2018) | Optimization | MobileNetV2 | Tiny ImageNet |
| 27 | EMA (Exponential Moving Average) | 1992 | Smooths weights, better final model | Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992) | Optimization | MobileNetV2 | Tiny ImageNet |
| 28 | LayerScale | 2021 | Per-channel scaling for stability | Touvron et al., "Going deeper with Image Transformers" (ICCV 2021) | Architecture | MobileNetV2 | Tiny ImageNet |
| 29 | ConvNeXt Block | 2022 | Combines all modern techniques | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Architecture | **ConvNeXt** |  Tiny ImageNet |
| 30 | Larger Kernels (7×7) | 2022 | Larger receptive field | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Architecture | ConvNeXt | Tiny ImageNet |
| 31 | Patchify Stem | 2022 | Efficient input processing | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Architecture | ConvNeXt | Tiny ImageNet |
| 32 | Layer-wise LR Decay | 2022 | Different LRs for different layers | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Training Strategy | ConvNeXt | **ImageNet-1k** |
| 33 | Global Response Normalization (GRN) | 2023 | Feature competition, reduces overfitting | Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (CVPR 2023) | Normalization | **ConvNeXt V2** | ImageNet-1k |
| 34 | Masked Autoencoder (MAE) Pretraining | 2023 | Self-supervised learning for ConvNeXt V2 | Woo et al., "ConvNeXt V2" (CVPR 2023) | Training Strategy | ConvNeXt v2 | ImageNet-1k |
| 35 | ConvNeXt V2 Block (with GRN) | 2023 | Modern CNN + GRN + MAE | Woo et al., "ConvNeXt V2" (CVPR 2023) | Architecture | ConvNeXt V2 | ImageNet-1k |


## Honorable techniques

| Technique | Reason for exclusion |
|-----------|----------------------|
| Dilated Convolutions | Not used in ConvNeXt; more for segmentation tasks |
| RandAugment | Advanced augmentation but MixUp/CutMix are more distinctive; ConvNeXt uses both: table space limited |



This is the journey from LeNet-5 to ConvNeXT. Each branch implements one improvement.