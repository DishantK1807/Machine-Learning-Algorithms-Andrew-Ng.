#K-Means Clustering

- It is a method of vector quantization popular for cluster analysis in data mining.
 
- k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 

- This results in a partitioning of the data space into *Voronoi cells*.

###Sample Implementation

- Using K-means Clustering to compress an image by reducing the number of colors to `k = 16` to represent the photo in a more efficient way by storing only the RGB values of the 16 colors present in the image

- Image Courtesy: [favim.com](http://favim.com/)

**K-means Algorithm**
- For initialization, sample 16 colors randomly from the original small picture. These are the `k means: u1, u2....uk` 

- Through each pixel in image, calculate its nearest mean.
```
c(i) := arg min ||x(i) - u(j)||^2
```

- Update the values of the means based on the pixels assigned to them. 
```
u(j) := sum{c(i)=j}x(i) / sum{c(i)=j}
```

- Repeat the above steps until convergence. It takes between 30 and 100 iterations. Either run the loop for a preset maximum number of iterations, or you can decide to terminate the loop when the locations of the means are no longer changing by a significant amount.
