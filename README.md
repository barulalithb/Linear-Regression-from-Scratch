# Linear-Regression-from-Scratch
Here you predict a certain value derived from some input using linear regression and for this, 
you need to understand the statistics behind Linear Regression and basic syntax of python3. Firstly,
let us understand linear regression with a toy example subsequently we'll get an enriched overview of how it works from scratch.
```
we divide the complete code into two classes,
  1. Linear Regression from Scratch
  2. Linear Regression from sklearn(Machine Learning Library in python).
```
```
Before getting into linear Regression, firstly let us understand the perspective in which data can be understood.
  1. Statistical Perspective of data
     output=funtion(input)
	          Or
     DependentVariable=f(IndependentVariable)
  2. Algorithmic view
     Model=Algorithm(Data)
     (Model:Specific Patterns Learned from Data
      Algorithm: The process of learning/understanding Data) 
```

So, here our model is Linear Regression and we write a certain algorithm from scratch which predicts outcomes for given specific inputs.

LINEAR REGRESSION:
This examines the relationship between two variables by determining the line of best fit
The essential requirements for this fit are 
 Slope: m
Intercept: c

y=m.x+c
n= number of values in x or y

Slope of the given regression line can be determined by the formula,

m=$frac{n.\sumxy-\sum x\sum y\}{n.\sumx^2-(\sumx)^2 }$ 
The intercept of the given regression line can be determined by the formula,
	
 c=mean(y)-m.mean(x)

