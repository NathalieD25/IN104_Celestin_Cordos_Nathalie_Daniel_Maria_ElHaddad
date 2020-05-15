##A mettre en dehors du programme principal
def method_comparison(d1,d2):

    rf=0 #compter le nombre de metrics ou la method random forest correspond mieux
    lgr=0 #compter le nombre de metrics ou la regression lineaire correspond mieux 
    for k in d1.keys()-{'class_mod'}-{'confusion'}-{'method name'}:
        if d1[k]>d2[k]:
            if d1['method name']=='linear regression':
                lgr+=1
            if d1['method name']=='random forest':
                rf+=1
        else:
            if d2['method name']=='linear regression':
                lgr+=1
            if d2['method name']=='random forest':
                rf+=1
    if lgr>rf:
        method='logistic regression'
    else:
        method='random forest'
    return method 
    
 ##Dans le programme principal
 
 ##A rajouter a l'interieur de la boucle for sur les storages 
#c1=0 ###compteur pour la methode logistic regression
#c2=0 ##compteur pour la methode Random forest
 #if method_comparison(model1[k],model2[k])=='logistic regression':
  #c1+=1
 #else:
 # c2+=1
 
 
 ####A rajouter a l'exterieur de la boucle 
 # if c1>c2:
#     print ("The logistic regression is better")
# if c2>c1:
#     print ("The random forest is better")

