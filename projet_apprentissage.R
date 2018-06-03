#                   GNU GENERAL PUBLIC LICENSE
#                       Version 3, 29 June 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
#                            Preamble
#
#  The GNU General Public License is a free, copyleft license for
#software and other kinds of works.
#
#  The licenses for most software and other practical works are designed
#to take away your freedom to share and change the works.  By contrast,
#the GNU General Public License is intended to guarantee your freedom to
#share and change all versions of a program--to make sure it remains free
#software for all its users.  We, the Free Software Foundation, use the
#GNU General Public License for most of our software; it applies also to
#any other work released this way by its authors.  You can apply it to
#your programs, too.
# 

# lecture des données
training <- read.csv("datas/train.csv")
validing <- read.csv("datas/valid.csv")

# librairies requises pour les algorithmes utilisés
install.packages("glmnet")
require(glmnet)
install.packages("dummies")
require(dummies)
install.packages("pROC")
require(pROC)
install.packages("ROCR")
require(ROCR)
install.packages("rpart")
require(rpart)
install.packages("rpart.plott")
require(rpart.plot)
install.packages("ggplot2")
require(ggplot2)
install.packages("ggthemes")
require(ggthemes)
install.packages("randomForest")
library(randomForest)
install.packages("xgboost")
library(xgboost)
insall.packages("e1071")
library(e1071)
install.packages("neuralnet")
library(neuralnet)

# Pour l'optimisation des variables
install.packages("caret")
library(caret)

# Pour utiliser pipelines
# %>% pipepline pour faire combinaison de fonctions (assez pratique) 
install.packages("dplyr")
library(dplyr)   


#### mise en forme des donnees ####

DF<-training

# supression des lignes apparamment inutiles
DF <- DF[,-(1)]
#DF <- DF[,-(14:16)]


# remplacer les valeurs inconnues par -99 ou la moyenne ou la médiane
# le plus efficace est par -99 au niveau du score
DF[is.na(DF)] = -99

#NAToMedian <- function(x) replace(x, is.na(x), median(x, na.rm = TRUE))
#DF=replace(DF, TRUE, lapply(DF, NAToMedian))

#na.colums<-names(which(sapply(DF,function(x) sum(is.na(x))>0)))
#for (c in na.colums) {
#  DF[which(is.na(DF[,c])),c]=median(DF[which(!is.na(DF[,c])),c])}

#transformer les caractères ininterprétables en nouvelles catégories + chiffres
#pour les modèles qui n'interprètes pas les caractères

DF<-dummy.data.frame(DF)


# séparation du jeu de training en 2 de manière aléatoire
# environ 30% sert à la validation intermédiaire
eval<-sample(1:dim(DF)[1],20000)
test=DF[eval,]
train=DF[-eval,]


####### regression Logistique ####### 


##### penalisation forward #####


modele.null = glm(factor(target)~1, family = binomial, data = train)
modele.full = glm(factor(target)~., family = binomial, data = train)

# "forward stepwise" pour le choix des variables
step(modele.null, scope = list (lower=modele.null, upper=modele.full), direction = "forward")


# variables les plus impactantes dans l'ordre : 
# var_23 + age_prospect + var_5 + var_7 + var_27 + sexe + var_18 + var_24 + var_22 + var_26 + var_16 + var_12 + departement + var_6 + var_9 + var_19 + var_28 + id + var_4 + var_32


##### fin penalisation #####

# version non pénalisée
#modTest0=glm(formula = target~., family = binomial, data = train)

modPenaliseTest1 = glm(formula = factor(target) ~ var_23 + age_prospect + var_5 + 
                         var_7 + var_27 + sexe + var_18 + var_24 + var_22 + var_26 + 
                         var_16 + var_12 + departement + var_6 + var_9 + var_19 + 
                         var_28 + id + var_4 + var_32, family = binomial, data = train)



modTest0.predict=predict.glm(modPenaliseTest1, type="response", newdata=test) 
table(ifelse(modTest0.predict>0.5, 1,0), as.factor(test$target)) 

r = roc(test$target, modTest0.predict) #target puis le predict en premier

plot(r)
r$auc
# Area under the curve: 0.6224

#### passage donnee valid ####



DFV<-validing
DFV[is.na(DFV)] = -99
DFV <- DFV[,-(1)]

#DFV<-dummy.data.frame(DFV)


modPenaliseTest2.predict=predict.glm(modPenaliseTest1, type="response", newdata=DFV) #donne un score, il permet de "classer" les r?sultats
str(modPenaliseTest2.predict)
DFV.predict<-data.frame(DFV$id, modPenaliseTest2.predict)


write.csv(DFV.predict, file = "Predict_GLM.csv", row.names = FALSE)




####### RandomForest  ####### 

DFArbre<-training

DFArbre <- DFArbre[,-(1)]

DFArbre[is.na(DFArbre)] = -99

evalarbre<-sample(1:dim(DFArbre)[1],20000)
testarbre=DFArbre[evalarbre,]
trainarbre=DFArbre[-evalarbre,]

# modele basique de référence
mod.rf = randomForest(target~., data = trainarbre, do.trace=20, mtry = 8)


#### opti perf ####

#en cas de prob avec le RF si la variable departement n'est pas donnee comme numerique
trainarbre$departement = as.numeric(trainarbre$departement)
testarbre$departement = as.numeric(testarbre$departement)


control = trainControl(method="repeatedcv", number=4, repeats=1, search="grid")

tunegrid = expand.grid(.mtry=c(1:15))
rf_gridsearch = train(factor(target)~., data=trainarbre, method="rf", tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)

plot(rf_gridsearch)
# D'apres le grid search, mtry optimal = 9 ou mtry = 8 (different selon les tests)


#### test de l'opti ####

mod.rf.tri = randomForest(target~., data = trainarbre, do.trace=20, ntree=100, mtry=9)
predictR2<-predict(mod.rf.tri, type='response', newdata=testarbre)
areaR<-roc(testarbre$target, predictR2)
plot(areaR)
areaR$auc



#### importance des variables - random forest #### 


importance = importance(mod.rf)
varImportance = data.frame(Variables = row.names(importance), 
                           Importance = round(importance[ ,'MeanDecreaseGini'],2))

# creer une variable de rang base sur l'importance 
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Utiliser ggplot pour visualiser l'importance des variables 
pdf("./resultats/variableImportance_randomforest.pdf") 
print(
  ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                             y = Importance, fill = Importance)) +
    geom_bar(stat='identity') + 
    geom_text(aes(x = Variables, y = 0.5, label = Rank),
              hjust=0, vjust=0.55, size = 4, colour = 'red') +
    labs(x = 'Variables') +
    coord_flip() + 
    theme_few()
)
dev.off() 



#### version pour donnees valid ####

DFVArbre<-validing

DFVArbre <- DFVArbre[,-(1)]
DFVArbre[is.na(DFVArbre)] = -99

predictR2<-predict(mod.rf.tri, type='response', newdata=DFVArbre)
str(predictR2)
DFVR.predict = data.frame(DFVArbre$id,predictR2)


# exporter dans un csv 
write.csv(DFVR.predict, file = "Predict_RF.csv", row.names = FALSE)





####### arbre Cart  ####### 

# methode un peu mediocre avec cet echantillon de donnee


DFVCart<-validing
DFCart<-training

evalcart<-sample(1:dim(DFCart)[1],20000)
testcart=DFCart[evalcart,]
traincart=DFCart[-evalcart,]

# calibrer le modele sur le training set 
mod.cart <- rpart(target ~ ., data = traincart, method = "class", cp=0.00001)
# cp: parametre de complexite : eviter overfit 

# regarder les resultats du modele calibre
summary(mod.cart)

# Elagage: trouver le parametre de complexite 
# Il est probable que l'arbre presente trop de feuilles pour une bonne prevision.
# Il est donc necessaire d'en reduire le nombre par elagage. C'est un travail delicat d'autant que la documentation n'est pas tres explicite et
#surtout les arbres des objets tres instables.

printcp(mod.cart)
plotcp(mod.cart)
# regarder la performance, en ordonnee: cp: prendre le cp qui optimise la performance 
# la performance a bcp augmente au debut (pente eleve au debut), puis stagne, 

mod.cart = prune(mod.cart,cp= 0.00043)



# visualiser l'arbre
prp(mod.cart)

# exporter en pdf
pdf("./resultats/arbre.pdf") 

print(prp(mod.cart))
dev.off() # fermer l'outil de dessin en pdf 

# visualiser les resultats
summary(mod.cart)
print(mod.cart)


# predire les etiquettes sur l'echantillon de test avec le modele calibre
pred.cart = predict(mod.cart, newdata = testcart, type = "class")

# utiliser la fonction prediction pour obtenir un resultat de prediction
pred.cart.rocr = prediction(as.numeric(pred.cart), as.numeric(testcart$target))

# utiliser la fonction performance pour obtenir une evaluation de la performance
perf.cart = performance(pred.cart.rocr, measure = "auc")
perf.cart.plot = performance(pred.cart.rocr, measure = "tpr", x.measure = "fpr")

# visualiser la courbe ROC avec la fonction plot 
plot(perf.cart.plot, colorize=T, main=paste("AUC:", (perf.cart@y.values)))

# le score auc 
auc.cart = as.numeric(perf.cart@y.values)
auc.cart 
# 0.5025822
# l'algorithme ne fonctionne pas meme boost => abandon du cart



###### xgboost ######



DFV<-validing
DF<-training

DF[is.na(DF)] = -99
DFV[is.na(DFV)] = -99
DF <- DF[,-(1)]
DFV <- DFV[,-(1)]

DF<-dummy.data.frame(DF)
DFV<-dummy.data.frame(DFV)

eval<-sample(1:dim(DF)[1],20000)
testxgb=DF[eval,]
trainxgb=DF[-eval,]



sparse_test = sparse.model.matrix(testxgb$target~., data = testxgb)
sparse_train = sparse.model.matrix(trainxgb$target~., data = trainxgb)
sparse_valid <- sparse.model.matrix(DFV$id~., data  = DFV)
dtrain = xgb.DMatrix(data = sparse_train, label = trainxgb$target)
dtest = xgb.DMatrix(data = sparse_test, label = testxgb$target)
watchlist <- list(train=dtrain, test=dtest)

######## debut opti param #######

bst <- xgb.train(data=dtrain, max_depth=4, eta=1, nthread = 2, nrounds=2, watchlist=watchlist, objective = "binary:logistic")

# cross validation = cv 
# trouver les hyperparametres optimises
# preciser cv 
cv.ctrl = trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                       allowParallel=T)

# preciser les parametres a tester 
xgb.grid = expand.grid(nrounds = 1000,
                       max_depth = c(2,4,6,8),
                       eta = c(0.01,0.05,0.1),
                       gamma = 1,
                       colsample_bytree = 1,  
                       min_child_weight = 1, 
                       subsample = 1
)
# tester 4 * 3 * 3 (cv) = 36 modeles a tester 
# les autres sont des parametre par defaut 
# site: xgboost documentation 

set.seed(100)
# train dans le package caret 
# tester les parametre
# "metric = objective" pour les r?gressions logistiques 
xgb_tune = train(target~.,
                 data=train,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="RMSE",
                 nthread =3
)
# nthread: utilsier combien de coeur d'ordinataeur 

print(xgb_tune)
# Les hyperparametres optimises sont: 
# la dernirere ligne 
# The final values used for the model were nrounds = 1000, max_depth = 8, eta = 0.05, gamma = 1, colsample_bytree = 1,
# min_child_weight = 1 and subsample = 1.


# construire le modele a partir de l'echantillon d'apprentissage
#my.label = as.numeric(levels(my_data_y_training))[my_data_y_training]

#dtrain = xgb.DMatrix(data = as.matrix(train), label = train$target)
# format particulier de xgboost 



# construire le modele xgboost
# les hyper parametres trouves en haut par train de caret

#mod2.xgb = xgboost(data = dtrain, nrounds = 300, max_depth = 8, eta = 0.05, gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample = 1, objective = "binary:logistic")


###### debut tests predicts #####

mod.xgb = xgboost(data = dtrain, nrounds = 1000, max_depth = 8, eta = 0.05, gamma = 1
          , colsample_bytree = 1, min_child_weight = 1
          , subsample = 1, objective = "binary:logistic")

# test avec moins de parametres
cv <- xgboost( data = sparse_train, label = trainxgb$target, max.depth = 8
               , nthread = 4, nround = 200, objective = "binary:logitraw")


pred <- predict(cv, sparse_test) 
pred1 <- predict(mod.xgb, dtest) #best pred

auc(testxgb$target, pred) 
auc(testxgb$target, pred1) #best auc

########## version pour valid ########## 


pred_valid <- predict(mod.xgb, sparse_valid)
str(pred_valid)
pred_valid.predict = data.frame(DFV$id,pred_valid)
write.csv(pred_valid.predict, file="Predict_XGBOOST.csv", row.names = FALSE)



# ****************************************
# ** importance des variables - xgboost ** 
# ****************************************
feature_name <- dimnames(sparse_train)[[2]]
importance_matrix <- xgb.importance(feature_name,
                                    model = mod.xgb
)
pdf("./resultats/variableImportance_xgboost.pdf") 
print(
  xgb.plot.importance(importance_matrix[1:20, ])
)
dev.off()

