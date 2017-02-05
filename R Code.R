##########################
#Problem 5: Insurance Data
##########################
library(glmnet)
library(corrplot)
library(MASS)

insurancedata = read.table("InsuranceData.txt", header=T)
head(insurancedata)

#pull out non-numeric fields and put the newpol column in front because it will
#be our "Y". This makes it easier to interpret the correlation matrices
insuranceNumeric = insurancedata[,c(6, 2:5, 8)]
head(insuranceNumeric)

#########Normal Linear Regression:
#[on a training set (30 out of 47 obs)]
lmFit = lm(newpol ~ ., data=insuranceNumeric[1:30, ])  # train on the first 30
summary(lmFit) 
plot(lmFit) #goes through several plots that test model assumptions

#explicitly create the training and test sets:
xTrain = insuranceNumeric[1:30, 2:6]
yTrain = insuranceNumeric[1:30, 1]
xTest = insuranceNumeric[31:47, 2:6]
yTest = insuranceNumeric[31:47, 1]

#How well does this model predict on the test data:
yHat = predict(lmFit, xTest) #feed xTest data into lmFit model to get predicted y values
yHat #confirm that you only have predicted y values for the test set (obs. 31:47)
rse = sqrt(sum((yTest - yHat)^2) / 24) #compute predicted rmse using degrees of freedom 
#specified in lm (lmFit)
rse
#visually compare a few of the predicted and actual values:
head(yHat)
head(yTrain) 




########################################
#Lasso Regularized Regression with no CV:
##########################################
#create training and test sets. we will use the training set to get lambda then get 
#the predicted rmse from the test set.
xTrain = insuranceNumeric[1:30, 2:6]
yTrain = insuranceNumeric[1:30, 1]
xTest = insuranceNumeric[31:47, 2:6]
yTest = insuranceNumeric[31:47, 1]

lassoFit = glmnet(as.matrix(xTrain), yTrain) #glmnet on the training data
lassoFit
plot(lassoFit, label=T) #plot shows you which betas are coming into be important and what 
#their coeff are. As lambda increases you get more and more var. contributions until all are 
#included

#here we can manually look for the best value of lambda
summary(lassoFit)  # not much help :(
print(lassoFit) # shows you # of variables involved in the model (indirectly via DF; 
# DF = n - #variables - 1);
# %dev column is like R^2 (% of variance explained); lambda value. Keep in mind that the model
#that explains the most (highest %dev) is not necessarily the best model, though for our 
#purposes here, we will assume it is. Here, it says lambda = 0.004723 is the best.



########################### 
#Lasso Regression with CV:
###########################
#Use lasso with cross validation on the training set to find an even better lambda:
cvFit <- cv.glmnet(as.matrix(xTrain), yTrain)
lassoFit = glmnet(as.matrix(xTrain), yTrain, lambda = cvFit$lambda.min, set.seed(523)) #glmnet on the 
#training data
lassoFit
# Let's look at the coefficients for the model at a specific lambda. We will choose the
#lambda chosen by CV (3 lines of code above):
coef(lassoFit, s=0.3108) #this gives us the coefficients for the variables included in the 
#model at a certain value of lambda. Note: s = lambda

#graph different values of lambda vs. rse:
par(mar=c(3, 3, 3, 3)) #sets the margins so you can see the graph better
plot(cvFit) #here we are looking at the log of the lambda. we are looking for what the
#coefficients are where we have the best lambda. we are looking for the lambda that gives the 
#minimum residual error on the crossvalidated sets (least prediction error), and we can ask
#R to find it for us (though it doesn't explicity tell us what the value is). 
#? this plot doesn't really match up with the optimized value of lambda chosen by CV above

#compute rse on the training dataset
yHat = predict(lassoFit, as.matrix(xTrain), s=0.3108) 
rse = sqrt(sum((yTrain - yHat)^2) / 24)
rse
#visually compare a few of the predicted and actual values:
head(yHat)
head(yTrain)

#compute rse on the test dataset (predicted rse)
yPredict = predict(lassoFit, as.matrix(xTest), s=0.3108)
rsePredict = sqrt(sum((yTest - yPredict)^2) / 24) 
rsePredict
#visually compare a few of the predicted and actual values:
head(yHat)
head(yTrain)


##################################################
#part 4: plot the residuals vs. predicted values
#################################################
resid = yTest - yPredict
yep<- data.frame(resid, yPredict)
yep
library(ggplot2)
library(gcookbook)
graph <- ggplot(yep, aes(x=yPredict, y=resid)) + geom_point() + theme_bw()
graph + xlab("Predicted Values") + ylab("Residuals") + geom_hline(yintercept=0)

plot(yPredict, resid, xlab="Predicted", ylab="Residuals")
abline(0, 0) #puts the horizon on the graph


########################################################
#Lasso Regularized Regression with CV in a different way:
#########################################################
#create training and test sets. we will use CV on the training set to get lambda then get 
#the predicted rmse from the test set.
xTrain = insuranceNumeric[1:30, 2:6]
yTrain = insuranceNumeric[1:30, 1]
xTest = insuranceNumeric[31:47, 2:6]
yTest = insuranceNumeric[31:47, 1]

# One of the nice things about this function is its 
# ability to test with cross validation to choose the lamba. Gets you an optimized lambda
cvfit = cv.glmnet(as.matrix(xTrain), yTrain) #cross-validated glmnet on the training data
cvfit
par(mar=c(3, 3, 3, 3)) #sets the margins so you can see the graph better
plot(cvfit) #here we are looking at the log of the lambda. we are looking for what the
#coefficients are where we have the best lambda. we are looking for the lambda that gives the 
#minimum residual error on the crossvalidated sets (least prediction error), and we can ask
#R to find it for us (though it doesn't explicity tell us what the value is) and give us the 
#model:
coef(cvfit, s="lambda.min")  # Note we are getting a lot more variable contributions here
#than we would expect given the correlation matrix

#we want to know which lambda gives the lowest rse on the predict set
yPredict = predict(cvfit, newx=as.matrix(xTest), s="lambda.min")
rsePredict = sqrt(sum((yTest - yPredict)^2) / 24)
rsePredict #the rse here (on the predict set) should be way lower than that given by the lm
#why isn't it?

head(insuranceNumeric)





######################################
#Problem 6: Employment Data
######################################
library(car)
library(corrplot)
library(ggplot2)
library(MASS)


employ =read.csv("EmploymentData.csv")
head(employ)

employNumeric = employ[2:10]
head(employNumeric)

# Check out the correlation of the original data
corM = cor(employNumeric)
corrplot(corM, method="ellipse") #note: multicollinearity is crazy strong
corrplot(corM, method="ellipse", order="AOE")
print(corM)
# Check out the covariance of the original data
covM = cov(employNumeric)
print(covM)

################# PCA:
p = princomp(employNumeric)
summary(p) 
plot(p)
abline(1,0, col="red") #allows you to notice which PCs have a SD above 1 for the cut-off

#princomp() calculates the rotated matrix for you so you don't have to multiply it out by
#hand:
p$scores
p$sdev
p$loadings #this is a matrix of the eigen vectors, and it tries to hide the things
#that aren't contributing

############# Rotate the components (PCA Factor Analysis):
library(psych)
p2 = psych::principal(employNumeric, rotate="varimax", nfactors=2, scores=TRUE)
print(p2$loadings, cutoff=.4, sort=F) #the goal of varimax is to get the loadings to be either 
#a 1 or a 0, amd a cutoff of .4 helps achieve this
print(p2$loadings, cutoff=.4, sort=T)
p2$loadings #note that this uses a much lower cutoff
p2$scores #look at new scores (how each sample scales along each PC). NOTE: these scores are 
#centered => they will always be around a mean of zero (so don't try to compare them directly
#to the data multiplied by the rotation matirx)
#The rows are in the same order as the original data so you can compare the original values to 
#their new scores on the PCs

#create a data frame that contains the countries and their new scores:
x = as.matrix(p2$scores) #first start with a matrix so you can reverse the signs
g = x*-1 #I am doing this because the sign is trivial, and it makes better interpretive sense
#if they are opposite of what R has defaulted them to
s = as.data.frame(g)
s$Country <-employ[, 1] #attach the country names as a field called 'Country'
s = s[, c(3, 1, 2)] #reorder them to put the country name first
s
attach(s) #allows you to call variables without naming the dataset first (e.g. RC1 vs. s$RC1)
#(adds the variable names to R's current session)
s[order(RC1),] #sort the dataframe rows by highest and lowest values of RC1
s[order(RC2),] #sort the dataframe rows by highest and lowest values of RC2

#See how original data values rank according to Agr, the primary component of RC1
head(employ)
attach(employ)
employ[order(Agr),]
#See how original data values rank according to Min, the primary component of RC2
employ[order(Min), c(1, 3, 2, 4:10)] #just reordered the columns so that Min is up front



# Run a correlation test to see which correlations are significant
library(psych)
options("scipen"=100, "digits"=5)
round(cor(employNumeric), 2) #gives you the correlation matrix; rounds it off to 2 decimals 
MCorrTest = corr.test(employNumeric, adjust="none") #this tests correlations for
#statistical significance. gives you a set of P values. adjust="none" => makes it so that the
# correlation matrix is perfectly symmetric
MCorrTest #remember, P < 0.1 is significant because we were asked to use a 90% C.L.

M = MCorrTest$p #this shows you the p values without rounding
M #need to use the un-rounded version for the MTest below:

# Now, for each element, see if it is < .1 (or whatever significance) and set the entry to 
# true = significant or false. keep in mind, we are running a massive number
#of T-tests here, and the chances of making a type I error are high, so it is better to be
# a bit more stringent and choose a high C.L.
MTest = ifelse(M < .1, T, F)
MTest 

# Now lets see how many significant correlations there are for each variable.  We can do
# this by summing the columns of the matrix
colSums(MTest) - 1  # We have to subtract 1 for the diagonal elements (self-correlation)
#we can use this to see if there are any variables that are overcorrelated (correlated with a 
#ton of other variables at a statistically significant level)

#Based upon the above test, Agr is collinear with more than 65% of the other predictors. This
#could be considered overcorrelation, and thus the variation seen in Agr may already be 
#accounted for in the other variables. This may justify its removal from the model, and we can
#see if removing it has the added bonus of making our PCs more easily interpretable:
head(employNumeric)
p3 = psych::principal(employNumeric[, 2:9], rotate="varimax", nfactors=2, scores=TRUE)
print(p3$loadings, cutoff=.4, sort=F) 



########################################
# Problem 7: Census Data
########################################
census =read.csv("CensusData.csv")
head(census)

################# PCA:
p = princomp(census)
summary(p) 
plot(p)
abline(1,0, col="red") #allows you to notice which PCs have a SD above 1 for the cut-off
p$loadings

#create a dataset with a scaled median home value:
census2 <- as.matrix(census)
smhv <- (census2[, 5] * 1/100000)
census2 = as.data.frame(census2)
census2$SMedianHomeVal <- smhv
census2 = census2[, c(1:4, 6)]
head(census2)

#PCA with scaled median home value:
p2 = princomp(census2)
summary(p2) 
plot(p2)
abline(1,0, col="red") #allows you to notice which PCs have a SD above 1 for the cut-off
p2$loadings

#PCA with correlation matrix (standardized values):
p3 = princomp(census, cor=T)
summary(p3) 
plot(p3)
abline(1,0, col="red") 
p3$loadings

head(p3$scores) #note how standarization makes the values all fall within a small and similar 
#range

#look at correlations
corrplot(cor(census), method = 'ellipse', order='AOE')

# Run a correlation test to see which correlations are significant
library(psych)
options("scipen"=100, "digits"=5)
round(cor(census), 2) #gives you the correlation matrix; rounds it off to 2 decimals 
MCorrTest = corr.test(census, adjust="none") #this tests correlations for
#statistical significance. gives you a set of P values. adjust="none" => makes it so that the
# correlation matrix is perfectly symmetric
MCorrTest #remember, P < 0.05 is significant because we were asked to use a 95% C.L.

M = MCorrTest$p #this shows you the p values without rounding
M #need to use the un-rounded version for the MTest below:

# Now, for each element, see if it is < .05 (or whatever significance) and set the entry to 
# true = significant or false. keep in mind, we are running a massive number
#of T-tests here, and the chances of making a type I error are high, so it is better to be
# a bit more stringent and choose a high C.L.
MTest = ifelse(M < .05, T, F)
MTest 

# Now lets see how many significant correlations there are for each variable.  We can do
# this by summing the columns of the matrix
colSums(MTest) - 1  # We have to subtract 1 for the diagonal elements (self-correlation)
#we can use this to see if there are any variables that are overcorrelated (correlated with a 
#ton of other variables at a statistically significant level)



##############################
# Problem 8: Track Record Data
##############################
track = read.table("TrackRecordData.txt", header=T)
head(track)

#transform hours to seconds:
trackseconds<- transform(track, m800=m800*60, m1500=m1500*60, m5000=m5000*60, m10000=m10000*60, 
                  Marathon=Marathon*60)
head(trackseconds)

tracknumeric = trackseconds[, 2:9]

#################################
#Non-Standardized Computations:

#PCA with covariance matrix (non-standardized values):
p = princomp(tracknumeric, cor=F)
summary(p) 
plot(p)
abline(1,0, col="red") 
p$loadings

#Rotate the components (PCA Factor Analysis):
library(psych)
pROTATED = psych::principal(tracknumeric, covar=TRUE, rotate="varimax", nfactors=2, 
                            scores=TRUE)
print(pROTATED$loadings, cutoff=.4, sort=F) #the goal of varimax is to get the loadings to be 
#either a 1 or a 0, and a cutoff of .4 helps achieve this
print(pROTATED$loadings, cutoff=.4, sort=T)
pROTATED$loadings #note that this uses a much lower cutoff


###############################
#Standardized Computations:

#PCA with correlation matrix (standardized values):
library(psych)
pstandardized = princomp(tracknumeric, cor=T)
summary(pstandardized) 
plot(pstandardized)
abline(1,0, col="red") 
pstandardized$loadings

#STANDARDIZED PCA Factor Analysis (PC rotation):
pstandardizedROTATED1 = psych::principal(tracknumeric, covar=FALSE, rotate="varimax", nfactors=1, 
                                        scores=TRUE)
print(pstandardizedROTATED1$loadings, cutoff=.4, sort=T) #the goal of varimax is to get the loadings to be either 
#a 1 or a 0, amd a cutoff of .4 helps achieve this
####################
pstandardizedROTATED2 = psych::principal(tracknumeric, covar=FALSE, rotate="varimax", nfactors=2, 
                                        scores=TRUE)
print(pstandardizedROTATED2$loadings, cutoff=.4, sort=T) #the goal of varimax is to get the loadings to be either 
#a 1 or a 0, amd a cutoff of .4 helps achieve this
#####################
pstandardizedROTATED3 = psych::principal(tracknumeric, covar=FALSE, rotate="varimax", nfactors=3, 
                                        scores=TRUE)
print(pstandardizedROTATED3$loadings, cutoff=.4, sort=T) #the goal of varimax is to get the loadings to be either 
#a 1 or a 0, amd a cutoff of .4 helps achieve this
####################
pstandardizedROTATED4 = psych::principal(tracknumeric, covar=FALSE, rotate="varimax", nfactors=4, 
                                        scores=TRUE)
print(pstandardizedROTATED4$loadings, cutoff=.4, sort=T) #the goal of varimax is to get the loadings to be either 
#a 1 or a 0, amd a cutoff of .4 helps achieve this
####################
pstandardizedROTATED5 = psych::principal(tracknumeric, covar=FALSE, rotate="varimax", nfactors=5, 
                                         scores=TRUE)
print(pstandardizedROTATED5$loadings, cutoff=.4, sort=T) #the goal of varimax is to get the loadings to be either 
#a 1 or a 0, amd a cutoff of .4 helps achieve this


#####################
pstandardizedROTATED3b = psych::principal(tracknumeric, covar=FALSE, rotate="varimax", nfactors=3, 
                                         scores=TRUE)
print(pstandardizedROTATED3b$loadings, cutoff=.7, sort=T) #use a cutoff of 0.7 instead



#####################
#Let's see if anything is over-correlated, and if it is, if removing it will 
#enhance the interpretibility of the PCs
####################
# Check out the correlation of the original data
corM = cor(tracknumeric)
corrplot(corM, method="ellipse") #note: multicollinearity is crazy strong
corrplot(corM, method="ellipse", order="AOE")
print(corM)
# Check out the covariance of the original data
covM = cov(tracknumeric)
print(covM)

# Run a correlation test to see which correlations are significant
library(psych)
options("scipen"=100, "digits"=5)
round(cor(tracknumeric), 2) #gives you the correlation matrix; rounds it off to 2 decimals 
MCorrTest = corr.test(tracknumeric, adjust="none") #this tests correlations for
#statistical significance. gives you a set of P values. adjust="none" => makes it so that the
# correlation matrix is perfectly symmetric
MCorrTest #remember, P < 0.01 is significant because we were asked to use a 99% C.L.

M = MCorrTest$p #this shows you the p values without rounding
M #need to use the un-rounded version for the MTest below:

# Now, for each element, see if it is < .01 (or whatever significance) and set the entry to 
# true = significant or false. keep in mind, we are running a massive number
#of T-tests here, and the chances of making a type I error are high, so it is better to be
# a bit more stringent and choose a high C.L.
MTest = ifelse(M < .01, T, F)
MTest 

# Now lets see how many significant correlations there are for each variable.  We can do
# this by summing the columns of the matrix
colSums(MTest) - 1  # We have to subtract 1 for the diagonal elements (self-correlation)
#we can use this to see if there are any variables that are overcorrelated (correlated with a 
#ton of other variables at a statistically significant level)

###########
#Try PCA with a few overcorrelated variables removed:
##########
track2 = tracknumeric[, c(1, 3:5)]
pstandardizedROTATEDa = psych::principal(track2, covar=FALSE, rotate="varimax", nfactors=1, 
                                         scores=TRUE)
print(pstandardizedROTATEDa$loadings, cutoff=.7, sort=T) 
############
pstandardizedROTATEDb = psych::principal(track2, covar=FALSE, rotate="varimax", nfactors=2, 
                                         scores=TRUE)
print(pstandardizedROTATEDb$loadings, cutoff=.7, sort=T) 
#############
pstandardizedROTATEDc = psych::principal(track2, covar=FALSE, rotate="varimax", nfactors=3, 
                                         scores=TRUE)
print(pstandardizedROTATEDc$loadings, cutoff=.7, sort=T) 
############
pstandardizedROTATEDd = psych::principal(track2, covar=FALSE, rotate="varimax", nfactors=4, 
                                         scores=TRUE)
print(pstandardizedROTATEDd$loadings, cutoff=.7, sort=T) 


##########################
#Common Factor Analysis:
##########################
cfa2 = factanal(tracknumeric, 2) #using 2 components
print(cfa2$loadings, cutoff=.7, sort=T) #note here that the % variance is calculated differently 
# in CFA so it won't be the same as it PCAFA, but they should be close.
summary(cfa2) #not helpful :(
#Common Factor Analysis doesn't have a nice plot like a scree plot :(
#################
cfa3 = factanal(tracknumeric, 3) 
print(cfa3$loadings, cutoff=.7, sort=T) 
#################
cfa4 = factanal(tracknumeric, 4) 
print(cfa4$loadings, cutoff=.7, sort=T) 
#################
cfa5 = factanal(tracknumeric, 5) 
print(cfa5$loadings, cutoff=.7, sort=T) 


