str(cars)

head(cars)
plot(cars)
m = lm(dist~speed, data = cars)
m
abline(m,col='red')
fitted(m)
residuals(m)
nx1 = data.frame(speed = c(21.5,25.0,25.5,26.0,26.5,27.0,27.5))
plot(nx1$speed,predict(m,nx1),col='red',cex = 2, pch = 20)
predict(m,nx1)
abline(m)


plot(cars,xlab ='speed',ylab='dist')
x = seq(0,25,length.out = 200)

for(i in 1:4){
  m = lm(dist~poly(speed,i),data = cars)
  assign(paste('m',i,sep='.'),m)
  lines(x,predict(m,data.frame(speed=x)), col=i)
}

anova(m.1,m.2,m.3,m.4)

str(women)
women_model = lm(weight~height,data=women)
coef(women_model)
plot(women)
abline(women_model,col='red')
summary(women_model)

install.packages("scatterplot3d")
library(scatterplot3d)

str(trees)
summary(trees)
?trees
scatterplot3d(trees$Girth,trees$Height,trees$Volume)
m = lm(Volume~Girth+Height,data = trees)
m
s = scatterplot3d(trees$Girth,trees$Height,trees$Volume,pch = 20, type='h',angle= 55)
s$plane3d(m)
ndata = data.frame(Girth=c(8.5,13.0,19.0),Height=c(72,86,85))
predict(m,newdata = ndata)

#decision tree model 
install.packages("rpart")
library(rpart)
r = rpart(Species~.,data = iris)
print(r)
par(mfrow = c(1,1), xpd = NA) # c(1,1)은 1,1 표에 display하겠다는 것.
plot(r)
text(r,use.n = TRUE)
#predict 함수를 사용하여 예측
p = predict(r,iris,type = 'class')
table(p,iris$Species) # confusion matrix
?rpart
#rpart 옵션중 사전확률 지정하는 옵션 예시

install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(r,type=4)

#decision tree의 장단점
#(-) : 성능이 낮음 (+) : 해석가능성(interpretability), 예측속도 빠름, 앙상블 기법을 사용하면 높은 성능
