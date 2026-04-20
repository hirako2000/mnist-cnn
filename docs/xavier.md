In 2010, Xavier Glorot and Yoshua Bengio published a paper. They derived a mathematical condition for initialization that preserves the variance of signals as they flow forward through the network and gradients as they flow backward. Their solution, now known as Xavier or Glorot initialization, sets the initial weights using a uniform distribution with bounds determined by the number of input and output connections of each layer.

For a layer with `fan_in` inputs and `fan_out` outputs, the weights are initialized from a uniform distribution in the range:

```
[-sqrt(6/(fan_in + fan_out)), +sqrt(6/(fan_in + fan_out))]
```

This ensures that the variance of the output of each layer equals the variance of its input, preventing signals from vanishing or exploding. Biases are initialized to zero.

## Benefits

Xavier initialization provides advantages over naive random initialization. Networks converge faster because they start from a better initial point in the loss landscape. Training becomes more stable and consistent across different random seeds, reducing the variance in final accuracy between runs. Most importantly, Xavier initialization enables the training of much deeper networks that would otherwise may fail to learn.

In our experiments with LeNet-5 on MNIST, the improvement over default (random) initialization was subtle but measurable. The Xavier-initialized model achieved a peak test accuracy of 99.19 percent, compared to 99.11 percent for the default initialization with ReLU. More notably, the model reached 99 percent accuracy by epoch nine, whereas the default initialization took ten epochs. The best test accuracy of 99.25 percent occurred at epoch thirteen, demonstrating more stable progression toward the peak.

However, we observed some drawbacks. The Xavier-initialized model showed some instability in later epochs, with test accuracy fluctuating more than the default initialization. This suggests that while Xavier provides a better starting point, the fixed learning rate of 0.001 becomes less appropriate as training progresses. The weights, initialized with larger variance, become more sensitive to gradient steps in later stages of training.

## Drawbacks

Xavier initialization was derived assuming tanh activation functions. For ReLU activations, which set half of their inputs to zero, the variance preservation condition changes. This led to the development of _He_ initialization in 2015, which specifically targets ReLU networks. For shallow networks like LeNet-5, the difference is minor, but for very deep networks, using Xavier with ReLU can still lead to signal decay.

A second limitation is that Xavier initialization assumes that all layers have the same activation variance properties. In practice, this is always true. The presence of pooling layers, batch normalization, and other modern techniques modifies signal propagation in ways that the original analysis did not account for.

Finally, Xavier initialization does not solve the problem of choosing appropriate learning rates. As we observed in our experiments, a well-initialized network can still diverge or plateau late in training if the learning rate remains constant. This is why learning rate scheduling, introduced later in the historical timeline, became a standard complement to proper initialization.

## Historical Impact

Xavier initialization was an enabler of the later deep learning discoveries. When AlexNet won ImageNet in 2012, it used a variant of this technique. The paper by Glorot and Bengio provided the theoretical foundation that allowed practitioner to confidently build deeper networks, knowing that initialization would not be the bottleneck.

For many years, Xavier initialization was the standard recommendation for networks with tanh or sigmoid activations. When ReLU became dominant, _He_ initialization replaced it, but the core insight—that initialization must preserve variance across layers—remained unchanged.

## What we changed

In our LeNet-5 implementation, we added an `_apply_xavier_init` method that iterates through all convolutional and fully connected layers and applies Xavier uniform initialization to the weights while setting biases to zero. The method is called automatically when the model is instantiated, ensuring consistent initialization without changing the training loop or any other aspect of the code.

The dataset remained identical. The optimizer, learning rate, and number of epochs were unchanged. This allowed us to isolate the effect of initialization and compare it directly against random  initialization.

## Results and Next Steps

The Xavier-initialized model achieved a best accuracy of 99.19 percent, outperforming both the default ReLU model (99.11 percent) and the original tanh model (98.99 percent). The model reached 99 percent accuracy by epoch nine, one epoch faster than the default initialization. However, we observed increased volatility in later epochs, suggesting that the fixed learning rate became mismatched with the weight scales late in training.

This observation points a later improvement in the journey: learning rate scheduling. By reducing the learning rate when progress stalls, we can stabilize late-stage training and potentially achieve even higher peak accuracy. However, historically, dropout (2012) came before learning rate scheduling as a widespread practice. Dropout addresses a different problem—overfitting—and operates orthogonally to initialization. This is what we will do next.

Xavier showed us that better initialization leads to faster convergence and higher peak accuracy but exposes the need for adaptive learning rates. Dropout will address overfitting, which remains visible in the gap between training and test accuracy. And learning rate scheduling will eventually tame the late-training instability we observed.

For a shallow network on a simple dataset, the benefits of Xavier initialization are slight but present. For the deeper networks, proper initialization is more noticible. Modern convolutional networks trains successfully with careful consideration of how its their start their training.