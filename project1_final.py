from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from imageio import imread
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn import metrics
from sklearn import linear_model
import random
import time as tm

#the franke function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#perform the OLS regression
def OLS (X,Xte,zt,Xn,trainn):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(zt)
    zpred = X.dot(beta)
    zpredtest = Xte.dot(beta)
    zzpred = np.reshape(Xn.dot(beta), (trainn,trainn))
    return beta, zpred, zpredtest, zzpred

#perform the ridge regression
def ridge (X,Xte,zt,lam,Xn,trainn):
    beta = np.linalg.inv(X.T.dot(X) + lam).dot(X.T).dot(zt)
    zpred = X.dot(beta)
    zpredtest = Xte.dot(beta)
    zzpred = np.reshape(Xn.dot(beta), (trainn,trainn))
    return beta, zpred, zpredtest,zzpred

#perform the lasso regression
def Lasso(X,Xte,zt,lamb,Xn,trainn):
    lasso=linear_model.Lasso(alpha=lamb)
    lasso.fit(X,zt)
    zpred = lasso.predict(X)
    zpredtest = lasso.predict(Xte)
    beta = lasso.coef_
    zzpred = np.reshape(lasso.predict(Xn), (trainn,trainn))
    return beta, zpred, zpredtest, zzpred

#function to calculate MSE and R2
#needs flattend z and zpred array
def MSER2 (z,zpred,n):

    ze = 0 #z error
    za = 0 #z sum of z - zavrage
    zm = np.mean(z) #calcute the mean of z
    #sum of error
    for i in range(0,n):
        ze = ze + (zpred[i] - z[i])**2
        za = za + (z[i] - zm)**2

    zMSE = ze/(n) #calcute MSE
    zR = 1 - (ze/za) #calcute R2

    return zMSE, zR

#creates an x matrix of the given polynomial
def Xmat (x,y,pol):
    size = len(x)

    if pol == 2:
        X  = np.c_[np.ones((size,1)), x, y, x**2, x*y,
        y**2]
    elif pol == 3:
        X  = np.c_[np.ones((size,1)), x, y, x**2, x*y,
        y**2, x**3, x**2*y, x*y**2, y**3]
    elif pol == 4:
        X  = np.c_[np.ones((size,1)), x, y, x**2, x*y,
        y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2,
        x*y**3, y**4]
    elif pol == 9:
        X  = np.c_[np.ones((size,1)), x, y, x**2, x*y,
        y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2,
        x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3,
        x*y**4, y**5, x**6, x**5*y, x**4*y**2, x**3*y**3,x**2*y**4, x*y**5, y**6,
        x**7, x**6*y, x**5*y**2, x**4*y**3,x**3*y**4, x**2*y**5, x*y**6, y**7,
        x**8, x**7*y, x**6*y**2, x**5*y**3,x**4*y**4, x**3*y**5, x**2*y**6, x*y**7,y**8,
        x**9, x**8*y, x**7*y**2, x**6*y**3,x**5*y**4, x**4*y**5, x**3*y**6, x**2*y**7,x*y**8, y**9]
    else:
        X  = np.c_[np.ones((size,1)), x, y, x**2, x*y,
        y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2,
        x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3,
        x*y**4, y**5]


    return X

#finds the variance of the beta parameters and calculate confidence interval
#also creates variance-covariance matrix
#needs flattend z and zpred array
def BetaConf (z,zpred,n,pred,X,beta):
    sigS = 0 #sigma squared of beta

    #sum of error
    for i in range(0,n):
        sigS = sigS + (z[i] - zpred[i])**2

    #calculate sigma squared
    sigS = sigS/((n) - pred - 1)
    sig = np.sqrt(sigS)

    #get the variance matrix
    XtXi = np.linalg.inv(X.T.dot(X))
    varb = XtXi * sigS

    #confidence interval
    #95% confidence interval givies a zscore of 1.96
    zscore = 1.96
    #zscore = 1.645
    #array to store confidence intervals
    conint = np.c_[np.zeros(X.shape[1]), np.zeros(X.shape[1])]

    #calculate the confidence interval
    for i in range(0,X.shape[1]):
        conint[i][0] = beta[i] - zscore*np.sqrt(XtXi[i][i])*sig
        conint[i][1] = beta[i] + zscore*np.sqrt(XtXi[i][i])*sig



    return sigS, conint, varb

#function to calculate Bias, variance and error terms of MSE
def VBE(zpred,z):

    #gets the mean for the prediced values of z
    zpm = np.mean(zpred)

    zv = 0
    zb = 0
    zer = 0
    if len(z) == len(zpred):
        n = len(z)

    for i in range(0,n):
        zv = zv + (zpred[i] - zpm)**2
        zb = zb + (z[i] - zpm)**2
        zer = zer + (z[i] - zpm)*(zpm - zpred[i])

    #variance calculation
    varz = zv / n
    #bias calculation
    biasz = zb / n
    #error calculation
    zeps = (2*zer) / n

    return varz, biasz, zeps


# Make data.
sampn = 100 #number of samples
lamb = 1 #value of lambda
trainp = 0.7 #Number of training samples given as %
polDeg = 3 #order of polynomial
bootrun = 1000 #times to run the bootrstap
method = "OLS"
nlevel = 0.1
#get number of training samples in each dimensjon
trainn = int(sampn*sampn*trainp)
#function for noise
N0 = np.random.normal(0,nlevel, (sampn,sampn))


#ordered x and y
# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
x = np.linspace(0,1,sampn)
y = np.linspace(0,1,sampn)

#random x and y
# x = np.random.rand(sampn,1)
# y = np.random.rand(sampn,1)


#meshgrid for x and y
xx, yy = np.meshgrid(x,y)


#z calculated using the franke function
#z = FrankeFunction(xx, yy) + N0

z = imread('SRTM_data_Norway_1.tif')
z = z[0:sampn, 0:sampn]

#LinearRegression
#Get all datapoints in one 1D array
xu = np.ravel(xx)
yu = np.ravel(yy)
zu = np.ravel(z)


#creates array to store data
zeps = np.zeros(bootrun)
zMSE = np.zeros(bootrun)
zR = np.zeros(bootrun)
zMSEte = np.zeros(bootrun)
zRte = np.zeros(bootrun)
timeend = np.zeros(bootrun)
varz = np.zeros(bootrun)
biasz = np.zeros(bootrun)

#gives a column size of the X matrix based on polynomial
matshape = 0
if polDeg == 2: matshape = 6
elif polDeg == 3: matshape = 10
elif polDeg == 4: matshape = 15
elif polDeg == 9: matshape = 55
else: matshape = 21

#creates values to find the best beta coefficients
minMSE = float("inf")
bestbeta = np.zeros(matshape)
bestconint = np.c_[np.zeros(matshape), np.zeros(matshape)]

#number of test data
testn = sampn*sampn - trainn


#get random indices for test data
# randi = random.sample(range(0,sampn*sampn),testn)
# f = open("testdata.txt", "w")
# for i in range(0,len(randi)):
#     f.write("%d "%randi[i])

#use indices from file
randi = np.loadtxt('testdata.txt',dtype='int',)


#array to store test data
xte = np.zeros(testn)
yte = np.zeros(testn)
zte = np.zeros(testn)

#get the test data from the full set
for k in range(0,testn):
    xte[k] = xu[randi[k]]
    yte[k] = yu[randi[k]]
    zte[k] = zu[randi[k]]

#remove the test data from the sample som they are not
#used for training
xuu = np.delete(xu,randi)
yuu = np.delete(yu,randi)
zuu = np.delete(zu,randi)


for j in range(0,bootrun):

    #set start time for the run
    timestart = tm.time()

    #get random indexes for the training data (with resampling)
    randi =np.random.randint((trainn), size=trainn)

    #array to store the trainging data
    xt = np.zeros(trainn)
    yt = np.zeros(trainn)
    zt = np.zeros(trainn)

    #takes trainging data with resampling
    for k in range(0,trainn):
        xt[k] = xuu[randi[k]]
        yt[k] = yuu[randi[k]]
        zt[k] = zuu[randi[k]]

    #create an X with x and y up to the fith polynomial
    X = Xmat (xt,yt,polDeg)

    #create X matrix for test data
    Xte = Xmat (xte,yte,polDeg)

    #creates new values for x and y to estimate and plot the prediction of z
    xn = np.linspace(0,1,sampn)
    yn = np.linspace(0,1,sampn)
    xxn, yyn = np.meshgrid(xn,yn)
    xn = np.ravel(xxn)
    yn = np.ravel(yyn)

    #make an X that has x and y that has increasing values
    Xn = Xmat (xn,yn,polDeg)

    #fit functions
    #ridge
    if method == "ridge":
        lam = np.identity(X.shape[1]) * lamb
        beta, zpred, zpredtest, zzpred = ridge(X,Xte,zt,lam,Xn,sampn)

    #lasso
    elif method == "lasso":
        beta, zpred, zpredtest,zzpred = Lasso(X,Xte,zt,lamb,Xn,sampn)

    #OLS
    else:
        beta, zpred, zpredtest, zzpred = OLS(X,Xte,zt,Xn,sampn)

    #calculate MSE and R2
    #needs flattend z and zpred array
    zMSE[j], zR[j] = MSER2(zt,zpred,len(zt))


    #calculate MSE and R2 for test data
    zMSEte[j], zRte[j] = MSER2(zte,zpredtest,len(zte))

    #function to calualte variance bias and error term
    varz[j], biasz[j], zeps[j] = VBE(zpredtest,zte)

    #gets the variance and confidence interval for the beta
    sigS, conint, varb = BetaConf(zt,zpred,trainn,matshape-1,X,beta)

    #stores the data from the model with the lowest MSE
    if(zMSEte[j] < minMSE):
        minMSE = zMSEte[j]
        bestbeta = beta
        bestconint = conint

    #gets the times used to make and use model
    timeend[j] = tm.time() - timestart



#emperical confidence intervals for the bias and variance
#calculates the lower and upper percentile based on alpha
alpha = 0.95
pu = ((1.0-alpha)/2.0) * 100
pl = (alpha+((1-alpha)/2)) * 100

#sort the array and get the upper an lower
#limits for the confidence intervals for
#the bias and variance
biasz = np.sort(biasz)
biaszl = np.percentile(biasz, pu)
biaszu = np.percentile(biasz, pl)

varz = np.sort(varz)
varzl = np.percentile(varz, pu)
varzu = np.percentile(varz, pl)



#prints releevant data
print("Polynomial of %d degree using the %s method and lambda of %f" %(polDeg, method, lamb))
print("Training data: MSE = %.3f   R2 = %.3f" %(np.mean(zMSE) , np.mean(zR)))
print("Test data:     MSE = %.3f   R2 = %.3f" %(np.mean(zMSEte) , np.mean(zRte)))
print("Bias average    : %f. 95 conf. interval: [%f,%f]" %(np.mean(biasz), biaszl,biaszu))
print("Variance average: %f. 95 conf. interval: [%f,%f]" %(np.mean(varz), varzl,varzu))
print("Time used by the program %.3f" %np.mean(timeend))
# for i in range(0,matshape):
#     print("%.3f & [%.3f,%.3f]\\\\" %(bestbeta[i],bestconint[i][0], bestconint[i][1]))




#plots figure 1: surface that gets estimated
fig = plt.figure(1)
ax = fig.gca(projection='3d')
plt.title("Surface of the real data", fontsize = 12)
ax.set_xlabel('x-axis',fontsize = 10)
ax.set_ylabel('y-axis',fontsize = 10)
ax.set_zlabel('z-axis',fontsize = 10)

ax.plot_surface(xx, yy, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

#plots figure 2: the estimated surface
fig = plt.figure(2)
ax = fig.gca(projection='3d')
plt.title("Fitted surface using the Lasso method and a polynomial of %d degree " %polDeg, fontsize = 12)
ax.set_xlabel('x-axis',fontsize = 10)
ax.set_ylabel('y-axis',fontsize = 10)
ax.set_zlabel('z-axis',fontsize = 10)
ax.plot_surface(xxn, yyn, zzpred, cmap=cm.coolwarm,linewidth=0, antialiased=False)


#plt.show()
