LeNet-5 was a groundbreaking architecture that proved [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_layer) could learn to recognize handwritten digits with remarkable accuracy. However, it used the hyperbolic tangent (tanh) as its [activation function](https://en.wikipedia.org/wiki/Activation_function). For nearly twelve years, tanh remained the standard choice for neural networks, despite a problem that would become increasingly impacting as networks grew deeper.

Tanh is a smooth, S-shaped function that maps any real number to an output between -1 and 1.

Its mathematical form is (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ). 
 
The shape of this curve means that for very large positive or very large negative inputs, the output saturates at either 1 or -1. In these saturated regions, the gradient of the function approaches zero. During training, when [gradients](https://en.wikipedia.org/wiki/Gradient) are [backpropagated](https://en.wikipedia.org/wiki/Backpropagation) through many layers, these small gradients get multiplied repeatedly. The result is that gradients in the early layers of a deep network effectively vanish to zero, preventing those layers from learning. This is the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).

In 1998, with only five layers in LeNet-5, the problem was manageable. But as researchers attempted to build deeper networks in the following years, they consistently found that adding more layers made training slower, harder, or impossible.

## Xavier

In 2010, Vinod Nair and [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) proposed an alternative: the [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectified_linear_unit), or ReLU. The function is simple: f(x) = max(0, x). It outputs zero for any negative input, and passes positive inputs through unchanged.

This means that gradients flow backward through the network without being attenuated. The vanishing gradient problem disappears. For the first time, researchers could stack many layers and still have  layers receive strong learning signals.

## Benefits

Switching from tanh to ReLU produced measurable improvements. Networks trained with ReLU typically converge two to three times faster than those using tanh, reaching the same accuracy in far fewer epochs. The computational cost is also lower: evaluating max(0, x) requires no exponential operations, making each forward and backward pass faster.

In our experiments with LeNet-5 on MNIST, the improvement seemed modest but clear. Over twenty epochs, the tanh version achieved a best test accuracy of 98.99 percent. The ReLU version reached 99.11 percent. More importantly, the test [loss](https://en.wikipedia.org/wiki/Loss_function) dropped by 23 percent, from 0.0448 to 0.0345, indicating that the ReLU model made its correct predictions with higher confidence. The convergence speed difference: ReLU reached ninety-nine percent accuracy ten epochs earlier than tanh.

Beyond these quantitative results, ReLU introduced sparsity to the network. Any neuron that receives a negative input outputs exactly zero, and those neurons do not contribute to subsequent computations. This natural sparsity means that for any given input, a significant fraction of the network is inactive, which may act as a form of regularization and reduces computational load.

## Drawbacks

ReLU is not without problems. One is the issue of dead neurons. If a neuron receives consistently negative inputs, its output will always be zero. The gradient through that neuron is also zero, so it will never update its weights to escape this state. Once a neuron dies, it stays dead. This is  problematic when using high learning rates, which can push many neurons into the dead region.

A second limitation is that ReLU is unbounded above. While tanh caps outputs at 1, ReLU can produce arbitrarily large values. This can sometimes lead to numerical instability or exploding activations in very deep networks, though this is less common than the vanishing gradient problem.

Several variants have been proposed to address these issues. Leaky ReLU allows a small, non-zero gradient for negative inputs. Parametric ReLU learns the slope of the negative part. Exponential Linear Units (ELUs) saturate on the negative side to push mean activations toward zero. However, the original ReLU remains widely used due to its simplicity and empirical performance.

## Historical impact

When [Alex Krizhevsky](https://en.wikipedia.org/wiki/Alex_Krizhevsky) built AlexNet for the 2012 ImageNet competition, he used ReLU and was able to train an eight-layer network that outperformed all previous approaches. AlexNet's victory is credited with sparking the deep learning revolution.

Today, ReLU and its variants are present in many modern convolutional neural network. The path from LeNet-5 to modern architectures like ConvNeXT passes directly through this simple, elegant idea: sometimes the best functions are the simplest.

## Repo changes & results

We replaced every tanh activation with ReLU. The forward function changed from applying tanh after each convolutional and fully connected layer to applying relu instead. No other aspect of the architecture was modified. The dataset remained exactly the same. The training procedure, including the optimizer, learning rate, and number of epochs, was unchanged. This allowed us to isolate the effect of the activation function and measure its impact directly.

<div align="center">
  <img src="https://raw.githubusercontent.com/hirako2000/mnist-cnn/improvement/relu/visualisations/training_comparison_relu_vs_tanh.avif" 
       alt="ReLU vs Tanh Comparison"
       width="90%">
  <br>
  <em>Training dynamics: ReLU (green) vs Tanh (red) over 20 epochs</em>
</div>


The result was a modest but measurable improvement in accuracy, a substantial reduction in test loss, and noticeably faster convergence. For a shallow network on a simple dataset, the benefits are incremental. For the deeper networks that would follow, ReLU was not just an improvement but a necessity.

