import matplotlib.pyplot as plt
from scipy import stats

X = [5,7,8,7,2,17,2,9,4,11,12,9,6]
Y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

'''slope: float, Slope of the regression line.
intercept: float, Intercept of the regression line.
rvalue: float, Correlation coefficient.
pvalue: float, Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with t-distribution of the test statistic.
stderr: float, Standard error of the estimated gradient.'''
slope, intercept, r, p, std_err = stats.linregress(X,Y)

plt.scatter(X,Y)

def predictfunc(x):
    return x * slope + intercept

prediction = list(map(predictfunc, X))
plt.plot(X,prediction)

plt.show()

print(r)