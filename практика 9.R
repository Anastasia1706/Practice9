library('e1071')     # SVM
library('ROCR')      # ROC-кривые
library('ISLR')      # данные по экспрессии генов

my.seed <- 1
data(Auto)
str(Auto)
high.mpg <- ifelse(Auto$mpg <= 23, 'blue', 'red')
# создаём наблюдения

x <- matrix(c(Auto$displacement, Auto$horsepower), ncol=2)
y <- c(high.mpg)

# таблица с данными, отклик -- фактор 
dat <- data.frame(x = x, y = as.factor(y))
plot(x, col = y, pch = 19)

train <- sample(1:nrow(dat), nrow(dat)/2) # обучающая выборка -- 50%

# SVM с радиальным ядром и маленьким cost
svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial", 
              coef0=2, cost = 1)
plot(svmfit, dat[train, ])

summary(svmfit)

# SVM с радиальным ядром и большим cost
svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial", 
              coef0=2, cost = 1e5)
plot(svmfit, dat[train, ])

# перекрёстная проверка
set.seed(1)
tune.out <- tune(svm, y ~ ., data = dat[train, ], kernel = "radial", 
                 ranges = list(cost = c( 0.1,1, 10, 100, 1000),
                               coef0 = c( 0.5, 1, 2, 3, 4)))
summary(tune.out)

# матрица неточностей для прогноза по лучшей модели
tbl <- table(true = dat[-train, "y"], 
      pred = predict(tune.out$best.model, newdata = dat[-train, ]))

# оценка точности
acc.test <- sum(diag(tbl))/sum(tbl)
acc.test


# функция построения ROC-кривой: pred -- прогноз, truth -- факт
rocplot <- function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

# последняя оптимальная модель
svmfit.opt <- svm(y ~ ., data = dat[train, ], 
                  kernel = "radial", coef0=0.5, cost = 10, decision.values = T)

# количественные модельные значения, на основе которых присваивается класс
fitted <- attributes(predict(svmfit.opt, dat[train, ],
                             decision.values = TRUE))$decision.values

# график для обучающей выборки
par(mfrow = c(1, 2))
rocplot(fitted, dat[train, "y"], main = "Training Data")


svmfit.flex = svm(y ~ ., data = dat[train, ], kernel = "radial", 
                  coef0=0.5, cost = 0.1, decision.values = T)
fitted <- attributes(predict(svmfit.flex, dat[train, ], 
                             decision.values = T))$decision.values
rocplot(fitted, dat[train,"y"], add = T, col = "red")

# график для тестовой выборки
fitted <- attributes(predict(svmfit.opt, dat[-train, ], 
                             decision.values = T))$decision.values
rocplot(fitted, dat[-train, "y"], main = "Test Data")
fitted <- attributes(predict(svmfit.flex, dat[-train, ], 
                             decision.values = T))$decision.values
rocplot(fitted, dat[-train, "y"], add = T, col = "red")

