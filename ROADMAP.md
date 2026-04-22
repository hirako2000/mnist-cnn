## LeNet-5 → ConvNeXT

| # | Technique / Architecture | Year | Benefit | Paper | Category | Milestone | Dataset |
|---|--------------------------|------|---------|-------|----------|-----------|---------|
| 1 | LeNet-5 (Baseline) | 1998 | Original convolutional architecture | [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (LeCun et al., 1998) | Architecture | **LeNet-5** | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 2 | ReLU Activation | 2010 | Solves vanishing gradient problem | [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) (Nair & Hinton, ICML 2010) | Activation | LeNet-5 | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 3 | Xavier Init | 2010 | Stable gradient variance | [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (Glorot & Bengio, AISTATS 2010) | Initialization | LeNet-5 | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 4 | Dropout | 2012 | Prevents overfitting | [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) (Hinton et al., 2012) | Regularization | LeNet-5 | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 5 | Max Pooling | 1980s/1990s | Preserves sharp features; popularized by AlexNet | [Cresceptron: A Self-organizing Neural Network](https://www.cse.msu.edu/~weng/research/CresceptronIJCNN1992.pdf) (Weng, Ahuja & Huang, 1992) improvement over [Neocognitron](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) (Fukushima, 1980) | Pooling | LeNet-5 | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 6 | Data Augmentation | 2012 | Better generalization | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (Alex Krizhevsky, 2012) | Data Processing | LeNet-5 | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 8 | Local Response Normalization (LRN) | 2012 | Channel-wise normalization (precursor of BN) | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (Krizhevsky et al., 2012) | Normalization | LeNet-5 | [MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| 9 | AlexNet Architecture | 2012 | Deeper, wider, 2-group conv | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (Krizhevsky et al., 2012) | Architecture | **AlexNet** | [**Tiny ImageNet**](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 10 | Global Average Pooling | 2013 | Replace FC layers | Lin et al., "Network In Network" (ICLR 2014, preprint 2013) | Pooling | AlexNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 11 | 3×3 Conv Stacking (VGG-style) | 2014 | Increased depth with small kernels | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (  Simonyan & Zisserman, 2014) | Architecture | VGG | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 12 | Inception Module (1×1 Conv) | 2014 | Bottleneck for channel reduction, reduces parameters | Szegedy et al., "Going Deeper with Convolutions" (GoogLeNet, CVPR 2015) | Architecture | GoogLeNET | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 13 | Gradient Clipping | 2013 | Prevents exploding gradients (primarily for RNNs) | Pascanu et al., "On the difficulty of training recurrent neural networks" (ICML 2013) | Training Strategy | GoogLeNET | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 14 | Learning Rate Scheduling | 2015 | Systematic LR reduction | Standard practice; popularized by AlexNet (2012) | Optimization | GoogLeNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 15 | Batch Normalization | 2015 | Faster training, higher LR | Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (ICML 2015) | Normalization | GoogLeNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 16 | He Init | 2015 | ReLU-specific initialization | He et al., "Delving Deep into Rectifiers" (ICCV 2015) | Initialization | GoogLeNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 17 | ResNet Architecture | 2015 | Enables very deep networks (152 layers) | He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016) | Architecture | GoogLeNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 18 | Residual Connections | 2015 | Enables very deep networks | He et al., "Deep Residual Learning for Image Recognition" (ResNet, CVPR 2016) | Architecture | **ResNet** | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 19 | Adam Optimizer | 2014 | Adaptive learning rates | Kingma & Ba, "Adam: A Method for Stochastic Optimization" (ICLR 2015) | Optimization | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 20 | Stochastic Depth | 2016 | Dropout for residual blocks | Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016) | Regularization | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 21 | Label Smoothing | 2016 | Prevents overconfidence | Szegedy et al., "Rethinking the Inception Architecture" (Inception-v3, CVPR 2016) | Regularization | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 22 | GELU Activation | 2016 | Smoother than ReLU | Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (arXiv 2016) | Activation | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 23 | Layer Normalization | 2016 | Better for smaller batches; adopted for CNNs in ConvNeXt | Ba et al., "Layer Normalization" (arXiv 2016) | Normalization | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 24 | Depthwise Convolution | 2017 | Separates spatial/channel mixing | Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions" (CVPR 2017) | Architecture | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 25 | AdamW (Weight Decay) | 2017 | Decoupled weight decay for better generalization | Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2017) | Optimization | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 26 | Grouped Convolutions (ResNeXt) | 2017 | Cardinality dimension for better representation | Xie et al., "Aggregated Residual Transformations for Deep Neural Networks" (ResNeXt, CVPR 2017) | Architecture | ResNet | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 27 | Cosine Annealing | 2017 | Better LR schedule with warm restarts | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017) | Optimization | **MobileNetV2-style** | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 28 | Inverted Bottleneck | 2018 | Expand → Conv → Project | Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018) | Architecture | MobileNetV2 | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 29 | Squeeze-and-Excitation (SE) | 2018 | Channel attention | Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018) | Architecture | MobileNetV2 | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 30 | Stochastic Weight Averaging (SWA) | 2018 | Flatter minima, better generalization | Izmailov et al., "Averaging Weights Leads to Wider Optima" (UAI 2018) | Optimization | MobileNetV2 | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 31 | EMA (Exponential Moving Average) | 1992 | Smooths weights, better final model | Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992) | Optimization | MobileNetV2 | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 32 | LayerScale | 2021 | Per-channel scaling for stability | Touvron et al., "Going deeper with Image Transformers" (ICCV 2021) | Architecture | MobileNetV2 | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 33 | ConvNeXt Block | 2022 | Combines all modern techniques | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Architecture | **ConvNeXt** |  [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 34 | Larger Kernels (7×7) | 2022 | Larger receptive field | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Architecture | ConvNeXt | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 35 | Patchify Stem | 2022 | Efficient input processing | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Architecture | ConvNeXt | [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) |
| 36 | Layer-wise LR Decay | 2022 | Different LRs for different layers | Liu et al., "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022) | Training Strategy | ConvNeXt | **ImageNet-1k** |
| 37 | Global Response Normalization (GRN) | 2023 | Feature competition, reduces overfitting | Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (CVPR 2023) | Normalization | **ConvNeXt V2** | ImageNet-1k |
| 38 | Masked Autoencoder (MAE) Pretraining | 2023 | Self-supervised learning for ConvNeXt V2 | Woo et al., "ConvNeXt V2" (CVPR 2023) | Training Strategy | ConvNeXt v2 | ImageNet-1k |
| 39 | ConvNeXt V2 Block (with GRN) | 2023 | Modern CNN + GRN + MAE | Woo et al., "ConvNeXt V2" (CVPR 2023) | Architecture | ConvNeXt V2 | ImageNet-1k |


## Honorable techniques

| Technique | Reason for exclusion |
|-----------|----------------------|
| Dilated Convolutions | Not used in ConvNeXt; more for segmentation tasks |
| MixUp / CutMix | Advanced data augmentation (mix samples) [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) (Zhang et al., ICLR 2017); [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899) (Yun et al., 2019) |
| RandAugment | Advanced augmentation but MixUp/CutMix are more distinctive; ConvNeXt uses both but table space is limited |

## Missing:
- Overlapping Pooling (used by AlexNet)
- Random Erasing (used in ConvNEXT)
- Linear Warmup	 (2017) stabilizes early training with high LR. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (arXiv 2017)	an Optimization used by ConvNEXT
- RandAugment in honorable should move to the list (in fact used in ConvNEXT)

This is the journey from LeNet-5 to ConvNeXT. Each branch implements one improvement.