#Support Vector Machine : Linear Classification
- In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data and recognize patterns, used for classification and regression analysis. 
- The training data are linearly separable i.e. two hyperplanes can be in a way that they separate the data and there are no points between them, and then try to maximize their distance. 

##Sample Implementation

###2-Dimensional / 2-Feature classification problem
- There are two classes of data (represented with blue and green in the corresponding plot) with a separation gap
- Parameter `C` in the SVM optimization problem is a positive cost factor that penalizes misclassified training examples. 
- Training data: `model = svmtrain(trainlabels, trainfeatures, '-s 0 -t 0 -c 1');`
  - The last string argument tells LIBSVM to train using the options
  - a. -s 0, SVM classification
  - b. -t 0, a linear kernel, because a linear decision boundary is required
  - c. -c 1, a cost factor of 1
- `C=1` makes the outlier, but the decision boundary seems like a reasonable fit.
- `C=100` makes the outlier classify correctly, but the decision boundary doesn't seem like a natural fit for the rest of the data.
- Conclusion: When cost penalty is large, the SVM algorithm tries very hard to avoid misclassifications. The tradeoff is that the algorithm gives less weight to producing a large separation margin.

###E-Mail Text classification problem
- Training a linear SVM model on each of the four training sets with  default `C` SVM value. 
- Testing is done on on set the named `email_test.txt.` via the `svmpredict` command.
- Sample run output: `Accuracy = 98.4615% (256/260) (classification)`
