#Regularization

*Implementing `Regularized Linear Regression` and `Regularized Logistic Regression`*

Regularization, in the field of machine learning, refers to a process of introducing additional information in order to solve an ill-posed problem or to prevent overfitting.

## Regularized Linear Regression
- Modified cost function to prevent overfitting by adding 'preference' for certain parameter values
- ` J(theta) = (1/2)*(y - theta*x')*(y-theta*x')' + (alpha)*theta*theta' `
- New parameter values is as `theta = y*x(x*x'+ alpha*I) ^ (-1)`
- Makes the problem well-posed for any degree
- 'Shrinks' the parameters towards zero
- Regularization is data independent

## Regularized Logistic Regression

