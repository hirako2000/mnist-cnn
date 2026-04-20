## LeNet-5 → ConvNeXT

| # | Technique | Year | Benefit | Paper |
|---|------------|------|-------------|-------|
| 1 | LeNet-5 (Baseline) | 1998 | Original convolutional architecture | [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (LeCun et al., 1998) |
| 2 | ReLU Activation | 2010 | Solves vanishing gradient problem | [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) (Nair & Hinton, ICML 2010) |
| 3 | Xavier Init | 2010 | Stable gradient variance | [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (Glorot & Bengio, AISTATS 2010) |
| 4 | Dropout | 2012 | Prevents overfitting | Hinton et al., "Improving neural networks by preventing co-adaptation of feature detectors" (arXiv 2012) |
| 5 | Max Pooling | 2012 | Preserves sharp features | Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, NIPS 2012) |
| 6 | Data Augmentation | 2012 | Better generalization | Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, NIPS 2012) |
| 7 | Gradient Clipping | 2013 | Prevents exploding gradients | Pascanu et al., "On the difficulty of training recurrent neural networks" (ICML 2013) |
| 8 | Adam Optimizer | 2014 | Adaptive learning rates | Kingma & Ba, "Adam: A Method for Stochastic Optimization" (ICLR 2015) |
| 9 | Learning Rate Scheduling | 2015 | Systematic LR reduction | Standard practice; popularized by AlexNet (2012) |
| 10 | Global Average Pooling | 2014 | Replace FC layers | Lin et al., "Network In Network" (ICLR 2014) |
| 11 | Batch Normalization | 2015 | Faster training, higher LR | Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (ICML 2015) |
| 12 | He Init | 2015 | ReLU-specific initialization | He et al., "Delving Deep into Rectifiers" (ICCV 2015) |
| 13 | Residual Connections | 2015 | Enables very deep networks | He et al., "Deep Residual Learning for Image Recognition" (ResNet, CVPR 2016) |
| 14 | Stochastic Depth | 2016 | Dropout for residual blocks | Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016) |
| 15 | Label Smoothing | 2016 | Prevents overconfidence | Szegedy et al., "Rethinking the Inception Architecture" (Inception-v3, CVPR 2016) |
| 16 | GELU Activation | 2016 | Smoother than ReLU | Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (arXiv 2016) |
| 17 | Layer Normalization | 2016 | Better for smaller batches | Ba et al., "Layer Normalization" (arXiv 2016) |
| 18 | Depthwise Convolution | 2017 | Separates spatial/channel mixing | Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions" (CVPR 2017) |
| 19 | Cosine Annealing | 2017 | Better LR schedule | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017) |
| 20 | Inverted Bottleneck | 2018 | Expand → Conv → Project | Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018) |
| 21 | Stochastic Weight Averaging (SWA) | 2018 | Flatter minima, better generalization | Izmailov et al., "Averaging Weights Leads to Wider Optima" (UAI 2018) |
| 22 | EMA (Exponential Moving Average) | 2018 | Smooths weights, better final model | Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992) / Common practice |
| 23 | LayerScale | 2021 | Per-channel scaling for stability | Touvron et al., "Going deeper with Image Transformers" (ICCV 2021) |
| 24 | Larger Kernels (7x7) | 2022 | Larger receptive field | Liu et al., "A ConvNet for the 2020s" (ConvNeXT, CVPR 2022) |
| 25 | Patchify Stem | 2022 | Efficient input processing | Liu et al., "A ConvNet for the 2020s" (ConvNeXT, CVPR 2022) |
| 26 | Layer-wise LR Decay | 2022 | Different LRs for different layers | Liu et al., "A ConvNet for the 2020s" (ConvNeXT, CVPR 2022) |
| 27 | ConvNeXT Block | 2022 | Combines all modern techniques | Liu et al., "A ConvNet for the 2020s" (ConvNeXT, CVPR 2022) |

This is the journey from LeNet-5 to ConvNeXT. Each branch implements one improvement.