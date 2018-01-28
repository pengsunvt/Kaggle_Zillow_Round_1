################################################################################################
#data process
library(CatEncoders)
library(ranger)
library(data.table)
library(Matrix)
library(dplyr)
setwd("C:/Users/peng/Desktop/Zillow")
train_2016=fread('input_data/train_2016_v2.csv',header=T)
train_2017=fread('input_data/train_2017.csv',header=T)
month=as.numeric(substr(train_2016$transactiondate,6,7))
tempy=train_2016$logerror[month>9]
index=seq(length(month))[month>9]
temp_month=month[month>9]
set.seed(123)
#sample twice to get the better validation sample
temp1=c(sample(index[temp_month==10],410,replace=F),
        sample(index[temp_month==11],720,replace=F),
        sample(index[temp_month==12],720,replace=F))
temp2=c(sample(index[temp_month==10],410,replace=F),
        sample(index[temp_month==11],720,replace=F),
        sample(index[temp_month==12],720,replace=F))
dev_mean1=abs(mean(train_2016$logerror[temp1])-mean(train_2016$logerror[month>9]))
dev_mean2=abs(mean(train_2016$logerror[temp2])-mean(train_2016$logerror[month>9]))
if(dev_mean1>=dev_mean2){
  index=temp2
}
if(dev_mean1<dev_mean2){
  index=temp1
}
index=as.numeric(index)
rm(list=ls()[ls()!="index"])
gc()
library(data.table)
library(Matrix)
library(dplyr)
setwd("C:/Users/peng/Desktop/Zillow")
sub=fread('input_data/sample_submission.csv',header=T)
temp=as.data.frame(sub[,1])  #fix the order of PID
names(temp)=c('parcelid')
test1=fread('input_data/properties_2016.csv',header=T)
test1=as.data.frame(test1)
test1=temp %>% left_join(test1, by = 'parcelid')
test2=fread('input_data/properties_2017.csv',header=T)
test2=as.data.frame(test2)
test2=temp %>% left_join(test2, by = 'parcelid')
p=dim(test1)[2]
for(i in 1:p){
  if(class(test1[,i])[1]=="integer" | class(test1[,i])[1]=="integer64"){
    test1[,i]=as.numeric(test1[,i])
    test2[,i]=as.numeric(test2[,i])
  }
  if(class(test1[,i])[1]=="character"){
    test1[,i]=as.numeric(as.factor(test1[,i]))
    test2[,i]=as.numeric(as.factor(test2[,i]))
  }
  test1[,i][is.na(test1[,i])]=median(test1[,i][!is.na(test1[,i])])
  test2[,i][is.na(test2[,i])]=median(test2[,i][!is.na(test2[,i])])
}
rm(list=c("i","p","temp"))
gc()
train1=fread('input_data/train_2016_v2.csv',header=T)
train1=train1 %>% left_join(test1, by = 'parcelid')
train1=as.data.frame(train1)
train2=fread('input_data/train_2017.csv',header=T)
train2=train2 %>% left_join(test2, by = 'parcelid')
train2=as.data.frame(train2)
train=rbind(train1,train2)
rm(train1);rm(train2);gc()
year=as.numeric(substr(train$transactiondate,1,4))
month=as.numeric(substr(train$transactiondate,6,7))
y=train$logerror
train=train %>% select(-c(transactiondate,logerror))
gc()
save.image("temporary_data/zillow_data.RData")
################################################################################################

################################################################################################
#xgboost model
library(xgboost)
library(data.table)
library(Matrix)
library(dplyr)
load("temporary_data/zillow_data.RData")
train_1=cbind(train,month)
x_train=train_1[(year==2017 & month>=3 & month<9) | year==2016,]
y_train=y[(year==2017 & month>=3 & month<9) | year==2016]
x_train=x_train[-index,]
y_train=y_train[-index]
x_valid1=train_1[year==2017 & month==9,]
y_valid1=y[year==2017 & month==9]
x_valid2=train_1[index,]
y_valid2=y[index]
yr=year[(year==2017 & month>=3 & month<9) | year==2016]
yr=yr[-index]
x_train=cbind(x_train,yr)
x_valid1$yr=2017
x_valid2$yr=2016
gc()
Pseudo_huber_obj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  x <- preds-labels
  grad <- x/sqrt(x^2+1)
  hess <- 1/(x^2+1)^(3/2)
  return(list(grad = grad, hess = hess))
}
train_3=x_train[y_train>-0.25 & y_train<0.3,]
y_3=y_train[y_train>-0.25 & y_train<0.3]
for(j in 2016:2017){
  for(i in 1:12){   
    y_3[train_3$month==i & train_3$yr==j]=y_3[train_3$month==i & train_3$yr==j]-
      median(y_3[train_3$month==i & train_3$yr==j])
  }
}
x_train=x_train %>% select(-c(censustractandblock,assessmentyear,month))
x_valid1=x_valid1 %>% select(-c(censustractandblock,assessmentyear,month))
x_valid2=x_valid2 %>% select(-c(censustractandblock,assessmentyear,month))
train_3=train_3 %>% select(-c(censustractandblock,assessmentyear,month))
dtrain=xgb.DMatrix(data.matrix(train_3), label=y_3)
dvalid1=xgb.DMatrix(data.matrix(x_valid1))
dvalid2=xgb.DMatrix(data.matrix(x_valid2))
set.seed(657)
param <- list(
  objective=Pseudo_huber_obj,
  eval_metric = "mae",
  eta = runif(1,0.001,0.1),
  lambda=runif(1,0.01,1.2), 
  alpha=runif(1,0,2),  
  max_depth=floor(runif(1,5,9)),
  base_score=mean(y_3),
  subsample=runif(1,0.1,0.9)
)
nr=floor(runif(1,100,350))
set.seed(657)
xgb_mod=xgb.train(data=dtrain,params=param,nrounds=nr)
xgb1=predict(xgb_mod,dvalid1) +0.0116
xgb2=predict(xgb_mod,dvalid2) +0.0118
s1=mean(abs(xgb1-y_valid1))
s2=mean(abs(xgb2-y_valid2))
x_test=test1%>%select(-c(censustractandblock,assessmentyear))
x_test$yr=2016
dtest=xgb.DMatrix(data.matrix(x_test))
pred0=predict(xgb_mod,dtest)+0.0118
x_test=test2%>%select(-c(censustractandblock,assessmentyear))
x_test$yr=2017
dtest=xgb.DMatrix(data.matrix(x_test))
pred1=predict(xgb_mod,dtest)+0.0118
sub=fread('input_data/sample_submission.csv', header = TRUE)
colnames(sub)[1] <- 'ParcelId'
sub$`201610`=round(pred0,4)
sub$`201611`=round(pred0,4)
sub$`201612`=round(pred0,4)
sub$`201710`=round(pred1,4)
sub$`201711`=round(pred1,4)
sub$`201712`=round(pred1,4)
write.csv(sub,"temporary_data/XGB.csv",row.names = FALSE)
gc()
rm(list=ls()[ls()!="xgb1" & ls()!="xgb2"])
save.image("temporary_data/XGB.RData")
gc()
################################################################################################

################################################################################################
#random forest model
library(dplyr)
library(ranger)
library(data.table)
library(Matrix)
load("temporary_data/zillow_data.RData")
train$dm=rep(0,length(train$parcelid))
train$dm[month<3 & month>8]=1
test1$dm=test2$dm=rep(1,length(test1$parcelid))
x_train=train[(year==2017 & month>=1 & month<9) | year==2016,]
y_train=y[(year==2017 & month>=1 & month<9) | year==2016]
x_train=x_train[-index,]
y_train=y_train[-index]
x_valid1=train[year==2017 & month==9,]
y_valid1=y[year==2017 & month==9]
x_valid2=train[index,]
y_valid2=y[index]
train_3=x_train[y_train>-0.1 & y_train<1,]
y_3=y_train[y_train>-0.1 & y_train<1]
x_train=x_train %>% select(-c(censustractandblock,assessmentyear))
x_valid1=x_valid1 %>% select(-c(censustractandblock,assessmentyear))
x_valid2=x_valid2 %>% select(-c(censustractandblock,assessmentyear))
train_3=train_3 %>% select(-c(censustractandblock,assessmentyear))
train_31=cbind(y_3,train_3)
set.seed(123)
fit1 <- ranger(y_3~.,data=train_31,importance="impurity",
               num.trees=200,mtry=8,min.node.size=150)
a1=predict(fit1,x_valid1,type='response')$predictions
mean(abs(a1-y_valid1))
a2=predict(fit1,x_valid2,type='response')$predictions
mean(abs(a2-y_valid2))
x_test1=test1 %>% select(-c(censustractandblock,assessmentyear))
pred11=predict(fit1,x_test1,type='response')$predictions
gc()
x_test1=test2 %>% select(-c(censustractandblock,assessmentyear))
pred21=predict(fit1,x_test1,type='response')$predictions
gc()
gc()
train_3=x_train[y_train<=(0.2) & y_train>-1,]
y_3=y_train[y_train<=(0.2) & y_train>-1]
train_31=cbind(y_3,train_3)
set.seed(123)
fit2 <- ranger(y_3~.,data=train_31,importance="impurity",
               num.trees=200,mtry=8,min.node.size=150)
b1=predict(fit2,x_valid1,type='response')$predictions
mean(abs(b1-y_valid1))
b2=predict(fit2,x_valid2,type='response')$predictions
mean(abs(b2-y_valid2))
x_test1=test1 %>% select(-c(censustractandblock,assessmentyear))
pred12=predict(fit2,x_test1,type='response')$predictions
gc()
x_test1=test2 %>% select(-c(censustractandblock,assessmentyear))
pred22=predict(fit2,x_test1,type='response')$predictions
rm(test1)
rm(test2)
gc()
rf1=0.376714895*a1+0.334079712*b1+0.003010825
mean(abs(rf1-y_valid1))
rf2=0.376714895*a2+0.334079712*b2+0.003010825
mean(abs(rf2-y_valid2))
pred1=0.376714895*pred11+0.334079712*pred12+0.003010825
pred2=0.376714895*pred21+0.334079712*pred22+0.003010825
colnames(sub)[1] <- 'ParcelId'
sub$`201610`=round(pred1,4)
sub$`201611`=round(pred1,4)
sub$`201612`=round(pred1,4)
sub$`201710`=round(pred2,4)
sub$`201711`=round(pred2,4)
sub$`201712`=round(pred2,4)
write.csv(sub,"temporary_data/RF.csv",row.names = FALSE)
gc()
rm(list=ls()[ls()!="rf1" & ls()!="rf2"])
save.image("temporary_data/RF.RData")
gc()
################################################################################################

################################################################################################
#genetic programming shared at: 
# https://www.kaggle.com/scirpus/genetic-programming-lb-0-0643904
system("python pre_defined_functions/Genetic_Programs_2016.py")
system("python pre_defined_functions/Genetic_Programs_2017.py")
################################################################################################

################################################################################################
#model ensembling
library(xgboost)
library(data.table)
library(Matrix)
library(dplyr)
setwd("C:/Users/peng/Desktop/Zillow")
load("temporary_data/zillow_data.RData")
rm(list=ls()[ls()!="index"])
#data import and process
library(data.table)
library(Matrix)
library(dplyr)
setwd("C:/Users/peng/Desktop/Zillow")
sub=fread('input_data/sample_submission.csv',header=T)
temp=as.data.frame(sub[,1])  #fix the order of PID
names(temp)=c('parcelid')
test1=fread('input_data/properties_2016.csv',header=T)
gp1=fread('temporary_data/GP2016.csv',header=T)
test1=as.data.frame(test1)
test1=temp %>% left_join(test1, by = 'parcelid')
test2=fread('input_data/properties_2017.csv',header=T)
gp2=fread('temporary_data/GP2017.csv',header=T)
names(gp1)=names(gp2)=c("i1","i2","i3","i4","i5")
test2=as.data.frame(test2)
test2=temp %>% left_join(test2, by = 'parcelid')
p=dim(test1)[2]
test1=cbind(test1,gp1)
test2=cbind(test2,gp2)
train1=fread('input_data/train_2016_v2.csv',header=T)
train1=train1 %>% left_join(test1, by = 'parcelid')
train1=as.data.frame(train1)
train1=train1[,c(1,2,3,61,62,63,64,65)]
train2=fread('input_data/train_2017.csv',header=T)
train2=train2 %>% left_join(test2, by = 'parcelid')
train2=as.data.frame(train2)
train2=train2[,c(1,2,3,61,62,63,64,65)]
train=rbind(train1,train2)
year=as.numeric(substr(train$transactiondate,1,4))
month=as.numeric(substr(train$transactiondate,6,7))
y=train$logerror
x_train=train[(year==2017 & month>=3 & month<9) | year==2016,]
y_train=y[(year==2017 & month>=3 & month<9) | year==2016]
x_train=x_train[-index,]
y_train=y_train[-index]
x_valid1=train[year==2017 & month==9,]
y_valid1=y[year==2017 & month==9]
x_valid2=train[index,]
y_valid2=y[index]
train_3=x_train[y_train>-0.4 & y_train<0.419,]
y_3=y_train[y_train>-0.4 & y_train<0.419]
gc()
fit=lm(y_3~i1+i2+i3+i4+i5,data=train_3)
gp1=predict.lm(fit,newdata=x_valid1)
mean(abs(gp1-y_valid1))
gp2=predict.lm(fit,newdata=x_valid2)
mean(abs(gp2-y_valid2))
pred1=predict.lm(fit,newdata=test1)
pred2=predict.lm(fit,newdata=test2)
y1=y_valid1
y2=y_valid2
rm(list=ls()[ls()!="y1" & ls()!="y2" & ls()!="gp1" & ls()!="gp2" & ls()!="pred1" &
               ls()!="pred2"])
load("C:/Users/peng/Desktop/Zillow/temporary_data/RF.RData")
load("C:/Users/peng/Desktop/Zillow/temporary_data/XGB.RData")
emf3=function(w){
  cv=0.4*mean(abs(w[1]*rf2+w[2]*xgb2+w[3]*gp2-y2))
  cv=cv+0.6*mean(abs(w[1]*rf1+w[2]*xgb1+w[3]*gp1-y1))
  return(cv)
}
re=optim(par=runif(3,0,1),fn=emf3,method="L-BFGS-B",lower=rep(0,3),upper=rep(1,3))
A=fread('temporary_data/RF.csv',header=T)
B=fread("temporary_data/XGB.csv",header=T)
pred2016=re$par[1]*A$`201610`+re$par[2]*B$`201610`+re$par[3]*pred1
pred2017=re$par[1]*A$`201710`+re$par[2]*B$`201710`+re$par[3]*pred2
sub=A
sub$`201610`=round(pred2016,4)
sub$`201611`=round(pred2016,4)
sub$`201612`=round(pred2016,4)
sub$`201710`=round(pred2017,4)
sub$`201711`=round(pred2017,4)
sub$`201712`=round(pred2017,4)
write.csv(sub,"temporary_data/sub1.csv",row.names = FALSE)
gc()
rm(list=ls())
gc()
################################################################################################

################################################################################################
#feature engineering for final model
library(CatEncoders)
library(data.table)
library(Matrix)
library(dplyr)
setwd("C:/Users/peng/Desktop/Zillow")
train_2016=fread('input_data/train_2016_v2.csv',header=T)
train_2017=fread('input_data/train_2017.csv',header=T)
submission=fread('input_data/sample_submission.csv',header=T)
test_2016=fread('input_data/properties_2016.csv',header=T)
test_2017=fread('input_data/properties_2017.csv',header=T)
gc()
data_clean=function(data){
  data=as.data.frame(data)
  p=dim(data)[2]
  for(i in 1:p){
    type=class(data[,i])
    if(type=="numeric" | type=="integer" | type=="integer64"){
      data[,i]=as.numeric(data[,i])
      data[,i][is.na(data[,i])]=-1
    }
    if(type=="character"){
      data[,i]=transform(LabelEncoder.fit(data[,i]),data[,i])
    }
  }
  gc()
  return (data)
}
pid=as.data.frame(submission$ParcelId)
names(pid)="parcelid"
test_2016=pid %>% left_join(test_2016, by = 'parcelid')
test_2017=pid %>% left_join(test_2017, by = 'parcelid')
test_2016=data_clean(test_2016)
test_2017=data_clean(test_2017)
fe=function(data,y){
  z1=rep(0,length(data[,1]))
  z1[data$regionidcounty==1286]=1
  z2=rep(0,length(data[,1]))
  z2[data$regionidcounty==3101]=1
  p7=rep(0,length(data[,1]))
  p7[data$pooltypeid7>0]=1
  p2=rep(0,length(data[,1]))
  p2[data$pooltypeid2>0]=1
  asy=rep(0,length(data[,1]))
  asy[data$assessmentyear==2016]=1
  ratio=data$taxamount/(data$structuretaxvaluedollarcnt+data$landtaxvaluedollarcnt)
  ratio[data$structuretaxvaluedollarcnt+data$landtaxvaluedollarcnt<=0]=0
  ratio[ratio<(-0.2)]=-0.2
  ratio[ratio>10]=10
  data=cbind(data,z1,z2,p7,p2,asy,ratio)
  avr=data$calculatedfinishedsquarefeet/(data$roomcnt+data$bedroomcnt+data$bathroomcnt)
  avr[(data$roomcnt+data$bedroomcnt+data$bathroomcnt)<=0]=0
  avr[data$calculatedfinishedsquarefeet<0]=0
  data=cbind(data,avr)
  ltx=log(data$taxamount+2)
  data=cbind(data,ltx)
  return (data)
}
test_2016=fe(test_2016)
test_2017=fe(test_2017)
test_2016=test_2016 %>% select(-c(assessmentyear))
test_2017=test_2017 %>% select(-c(assessmentyear))
rm(list=ls()[ls()!="test_2016" & ls()!="test_2017"])
gc()
save.image("C:/Users/peng/Desktop/Zillow/temporary_data/fe_data.RData")
################################################################################################

################################################################################################
#final model to deal with repeat sale
library(ranger)
library(data.table)
library(Matrix)
library(dplyr)
library(xgboost)
setwd("C:/Users/peng/Desktop/Zillow")
load("temporary_data/fe_data.RData")
train_2016=fread('input_data/train_2016_v2.csv',header=T)
train_2017=fread('input_data/train_2017.csv',header=T)
train_2016=train_2016 %>% left_join(test_2016, by = 'parcelid')
train_2017=train_2017 %>% left_join(test_2017, by = 'parcelid')
y=c(train_2016$logerror,train_2017$logerror)
day=as.numeric(as.Date(c(train_2016$transactiondate,train_2017$transactiondate)))-16800
month=as.numeric(substr(train_2017$transactiondate,6,7))+12
month=c(as.numeric(substr(train_2016$transactiondate,6,7)),month)
pdall=c(train_2016$parcelid,train_2017$parcelid)
#pdall=c(pdall,test_2016$parcelid[testv])
#month=c(month,rep(10,25))
A=fread('temporary_data/sub1.csv',header=T)
names(A)[1]="parcelid"
B=train_2017 %>% left_join(A, by = 'parcelid')
yp1=yp=B$`201710`
y1=B$logerror
mn1=month[90276:167888]-12
r17=rep(0,length(mn1))
for(j in 1:9){
  tempd=train_2017$parcelid[mn1==j]
  rtemp=rep(0,length(tempd))
  if(j<=3){
    rtemp[tempd %in% pdall[month<=(j+10)]]=1
  }
  if(j>=4){
    rtemp[tempd %in% pdall[month<=(j+10) & month>j-3]]=1
  }
  r17[mn1==j]=rtemp
}
set.seed(1)
temp_min=-100
for(iter in 1:400){
  col_num=floor(runif(1,15,30))
  col_select=sample(seq(1:29),col_num,replace=F)
  gv=rep(0,9)
  for(j in 1:9){
    X=train_2017[mn1!=j & r17==1,c(1,seq(4,32,1))]
    X=X[,-27]
    #names(X)[dim(X)[2]]="r17"
    Xv=train_2017[mn1==j & r17==1,c(1,seq(4,32,1))]
    Xv=Xv[,-27]
    #names(Xv)[dim(Xv)[2]]="r17"
    y0=y1[mn1!=j & r17==1]
    yv=y1[mn1==j & r17==1]
    fit1=lm(y0[abs(y0)<0.4]~.,data=X[abs(y0)<0.4,col_select])
    yp=predict.lm(fit1,newdata=Xv)
    gv[j]=(sum(abs(yv+0.007))-sum(abs(yv-0.5*yp+0.007*0.5)))/length(y1[mn1==j])
  }
  temp_score=min(gv)
  if(temp_score>temp_min){
    sol=col_select
    temp_min=temp_score
    print(temp_score)
  }
}
#check cross validation and make prediction
rep_sep=test_2017$parcelid[test_2017$parcelid %in% pdall[month<=19 & month>6]]
rep_oct=test_2017$parcelid[test_2017$parcelid %in% pdall[month<=20 & month>7]]
rep_nov=test_2017$parcelid[test_2017$parcelid %in% pdall[month<=21 & month>8]]
rep_dec=test_2017$parcelid[test_2017$parcelid %in% pdall[month<=21 & month>8]]
y_sep=rep(0,length(rep_sep))
y_oct=rep(0,length(rep_oct))
y_nov=rep(0,length(rep_nov))
y_dec=rep(0,length(rep_dec))
for(j in 1:9){
  X=train_2017[mn1!=j & r17==1,c(1,seq(4,32,1))]
  X=X[,-27]
  #names(X)[dim(X)[2]]="r17"
  Xv=train_2017[mn1==j & r17==1,c(1,seq(4,32,1))]
  Xv=Xv[,-27]
  #names(Xv)[dim(Xv)[2]]="r17"
  y0=y1[mn1!=j & r17==1]
  yv=y1[mn1==j & r17==1]
  fit1=lm(y0[abs(y0)<0.4]~.,data=X[abs(y0)<0.4,sol])
  yp=predict.lm(fit1,newdata=Xv)
  gv[j]=(sum(abs(yv-yp1[mn1==j & r17==1]))-sum(abs(yv-0.5*yp+0.007*0.5)))/length(y1[mn1==j])
  y_sep=y_sep+as.vector(predict.lm(fit1,newdata=test_2017[(test_2017$parcelid %in% rep_sep),]))
  y_oct=y_oct+as.vector(predict.lm(fit1,newdata=test_2017[(test_2017$parcelid %in% rep_oct),]))
  y_nov=y_nov+as.vector(predict.lm(fit1,newdata=test_2017[(test_2017$parcelid %in% rep_nov),]))
  y_dec=y_dec+as.vector(predict.lm(fit1,newdata=test_2017[(test_2017$parcelid %in% rep_dec),]))
}
y_sep=y_sep/9*0.5-0.5*0.007
y_oct=y_oct/9*0.5-0.5*0.007
y_nov=y_nov/9*0.5-0.5*0.007
y_dec=y_dec/9*0.5-0.5*0.007
y_sep[abs(y_sep)>0.1]=-0.007
y_oct[abs(y_oct)>0.1]=-0.007
y_nov[abs(y_nov)>0.1]=-0.007
y_dec[abs(y_dec)>0.1]=-0.007
sub1=fread('temporary_data/sub1.csv',header=T)
sub1$`201710`[sub1$ParcelId %in% rep_oct]=y_oct
sub1$`201711`[sub1$ParcelId %in% rep_nov]=y_nov
sub1$`201712`[sub1$ParcelId %in% rep_dec]=y_dec
write.csv(sub1,"output_submission/sub1.csv",row.names = FALSE)
gc()
rm(list=ls())
file.remove(paste("temporary_data/",dir("temporary_data"),sep=""))
gc()
################################################################################################