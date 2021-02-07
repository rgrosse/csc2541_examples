## K-FAC Example

This is a JAX reimplementation of some of the experiments from Martens and Grosse (2015), [Optimizing neural networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671), in particular the MNIST and Curves autoencoders.

This code is provided for pedagogical purposes to my course [CSC2541: Neural Net Training Dynamics](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/). I do not recommend it for use in production or for rigorous empirical comparisons. This only includes the block diagonal (not block tridiagonal) version of K-FAC, and there are likely other differences as well. See [the original MATLAB code](https://www.cs.toronto.edu/~jmartens/docs/KFAC3-MATLAB.zip) for a faithful implementation of the original experiments.

To run the MNIST autoencoder experiment:
```
python mnist.py
```

To run the Curves autoexperiment, first download the [Curves dataset](https://www.cs.toronto.edu/~jmartens/digs3pts_1.mat) to the code directory, and then run:
```
python curves.py
```
