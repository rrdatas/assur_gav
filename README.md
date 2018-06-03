# assur_gav
Une compagnie d’assurance souhaite construire un score d’appétence pour une « garantie accident de la vie », dans un contexte de vente croisée
Les données sont issues de diverses filiales du groupe : autres produits détenus, éléments socio-démographiques, habitudes de vie ou de consommation ainsi que la sinistralité observée.
Les opérations suivantes ont été réalisées sur les données :
- Suppression des catégories rares
- Discrétisation des variables numériques par quantile
- Pas de dimension de temps
L’appétence est désignée par la colonne « target », binaire. Il faut donc prédire les lignes avec une target à 1.

A réaliser
1) Prédire le comportement client sur le jeu de Validation 
- 3 prédictions minimum sur le jeu de validation
- Pour chacune de ces prédictions, livrer
o Un fichier csv
▪ Nom Predict_Modele.csv (respectez la casse)
▪ 2 colonnes (pas 3 .. Vérifiez)
▪ Garder le «. » comme séparation décimale
o Le code
o Visualisations du comportement associé à ces variables importantes (quel est leur impact sur target ?)