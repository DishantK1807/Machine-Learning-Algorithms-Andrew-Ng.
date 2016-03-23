#Non-linear SVM classification with kernels

- Sometimes, Classes may not be separable by a linear boundary which lead to implementaion of Nonlinear Classification using kernels.
- Here, an RBF kernel is used to classify data that is not linearly separable.

**Kernel:** 
  - In machine learning, kernel methods are a class of algorithms for pattern analysis, whose best known member is the support vector machine (SVM). 
  - Linearly non-separable features often become linearly separable after they are mapped to a high dimensional feature space. 
  - Kernels of feature mapping are easier to compute. 
  - Therefore, it's possible to create a very complex decision boundary based on a high dimensional (even infinite dimensional) feature mapping but still have an efficient computation because of the kernel representation.
  

**Radial Basis Function (RBF) kernel:** 
  - The Gaussian RBF kernel is a popular kernel function used in various kernelized learning algorithms. 
  - In particular, it is commonly used in support vector machine classification.
  
```
K(x(i),x(j)) = phi(x(i))'*phi(x(j))
             = exp(-(gamma) || x(i)-x(j) || ^ (2))
where (gamma) > 0
```

##Sample Implementation
- Accuracy = 99.7683% (861/863) (classification)

