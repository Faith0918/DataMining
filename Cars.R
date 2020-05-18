library(MASS)
head(Cars93)
?Cars93
for(i in 1:27){
  print(i)
  print(table(is.na(Cars93[,i])))
}

mean(Cars93$Luggage.room, na.rm=T)*28.3
boxplot(cars$dist)$stat
mean(ifelse(cars$dist<2|cars$dist>93,NA, cars$dist),na.rm=T)
