# -*- coding: utf-8 -*-
"""Projet ODATA par Narjisse LASRI et Estelle DUHEM"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns


from functions import *


def part3():
    pd.set_option('display.max_columns', None)
    
    
    #--> Importation des données :
    data = pd.read_csv('data.csv',engine = 'python')
    print("Voici les données : \n", data)

    #_________3.1 Examen des données_________
    
    print("\n\n_______3.1 Examen des données_______\n")
    print("Taille du jeu de données :")
    print(data.shape[0], " lignes et ", data.shape[1], " colonnes")
    print("\nType des données :")
    print(data.dtypes)
    print("\nNombre de valeurs manquantes : ", data.isna().sum().sum())
    print("\nMéthode describe :", data.describe())
    #afficher_histogrammes(data)
    
    wait = input("****** Appuyez sur ENTREE pour passer à la partie 3.2 Préparation des données")
       
    #_________3.2 Préparation des données_________
    
    print("\n\n_______3.2 Préparation des données_______\n")
    
    print("Nous allons rechercher les données aberrantes en suivant plusieurs méthodes.\n")
    print("---> 1ere methode, nous analysons au cas par cas les valeurs extrêmes d'après les histogrammes :\n")

    print("-Valeur max qui semble aberrante pour le GDP est celle du pays :", data.iloc[data['GDP'].argmax(),0], "avec la valeur : ", data['GDP'].max())
    print(" -> Après recherche cette valeur n'est pas aberrante, elle est correcte.\n")

    print("-Valeur max qui semble aberrante pour child_mortality est celle du pays :", data.iloc[data['child_mortality'].argmax(),0],"avec la valeur : ", data['child_mortality'].max())
    print(" -> Après recherche cette valeur n'est pas aberrante, elle est correcte.\n")
    
    print("-Valeur min qui semble aberrante pour life_expectation est celle du pays :", data.iloc[data['life_expectation'].argmin(),0],"avec la valeur : ", data['life_expectation'].min())
    print(" -> Après recherche la valeur correcte pour l'espèrance de vie du Bangladesh est 72.05, nous corrigeons.")
    index_bangladesh = data[data['country']=='Bangladesh'].index.values
    data.iloc[index_bangladesh, 7] = 72.05
    print("Valeur espérance de vie du bangladesh après modification : ", data.iloc[index_bangladesh, 7] )
    print("\n -> Nous détectons que l'espèrance de vie de Haiti est de 31.1 dans nos données, ce qui est aberrant, nous corrigeons par la vraie valeur qui est 63.29.")
    index_haiti = data[data['country']=='Haiti'].index.values
    data.iloc[index_haiti, 7] = 63.29
    print("Valeur espérance de vie de Haiti après modification : ", data.iloc[index_haiti, 7] )
    
    print("\n\n---> 2eme methode, nous appliquons la règle des 3 sigmas sur les attributs qui semblent suivre une loi normale (exports, imports et health):\n\n")

    
    print("Dans un premier temps, nous identifions les valeurs aberrantes et nous les remplaçons par NaN pour ensuite pouvoir les traiter en même temps que les valeurs manquantes.")
    data, column_country = remplacer_val_aberrantes(data)
    print("\nNombre de valeurs manquantes avec ajout des valeurs aberrantes : ", data.isna().sum().sum())
    print("\n*** Notre fonction remplacer_val_aberrantes a supprimé la colonne 'country', elle nous la renvoie donc pour que nous puissions travailler dessus plus tard")

    print("\nDans un second temps, nous remplaçons toutes les valeurs manquantes par la moyenne de l'attribut correspondant.")
    data = remplacer_val_manquantes(data, "moyenne")
    print("Nombre de valeurs manquantes après remplacement de celles-ci : ", data.isna().sum().sum())
    
    print("\n\nNous allons centrer et réduire nos données :")
    
    print("*** Nous devons travailler sur nos données sans la première colonne 'country' car elle ne contient pas de valeurs numériques.")
    
    
    X = data.to_numpy()

    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    Z = scaler.transform(X)

    print("Moyenne = ", Z.mean())
    print("Variance = ", Z.var())
    
    wait = input("****** Appuyez sur ENTREE pour passer à la partie 3.3 Recherche de corrélations")
    
    #_________3.3 Recherche de corrélations_________
    
    print("\n\n_______3.3 Recherche de corrélations_______\n")
    
    print("\nLes attributs corrélés sont : \n", liste_corr(data, 0.8))
    
    attributes_corr, max = liste_corr_max(data, "max")
    
    corr_matrix = data.corr(method='spearman')
    
    attributes_corr1, max1 = liste_corr_max(data, "max")
    
    print("\nLes attributs les plus corrélés sont : \n", attributes_corr1, "avec comme corrélation", max1)
    
    attributes_corr2, max2 = liste_corr_max(data, "positif")
    
    print("\nLes attributs les plus corrélés positivement sont : \n", attributes_corr2, "avec comme corrélation", max2)
    
    attributes_corr3, max3 = liste_corr_max(data, "negatif")
    
    print("\nLes attributs les plus corrélés négativement sont : \n", attributes_corr3, "avec comme corrélation", max3)

    
    attributes_corr3, max3 = liste_corr_max(data, "autre")
    
    print("\nLes attributs les moins corrélés sont : \n", attributes_corr3, "avec comme corrélation", max3)

    print("\nMatrice de corrélation :\n", corr_matrix)
    
    pd.plotting.scatter_matrix(corr_matrix, figsize=(10, 10))
    
    figE, axE = plt.subplots()
    axE=sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns.values, yticklabels=corr_matrix.columns.values)
    axE.set(title = "* Correlation coefficients *")
    
    wait = input("****** Appuyez sur ENTREE pour passer à la partie 3.4 Clustering des données")
    
    #_________3.4 Clustering des données_________
    
    print("\n\n_______3.4 Clustering des données_______\n")
    
    #-------------Methode KMeans---------------------------------------------
    print("\nNous cherchons le nombre de clusters idéal pour la méthode Kmeans. \nPour cela nous utilisons la méthode elbow et la méthode du score silhouette en fonction de k.")
    elbow_method_kmeans(Z)
    methode_silhouette_score(Z, 'kmeans')
    
    print("\nVoici les clusters obtenus pour la méthode k-means :\n")
    labels_Kmeans = apply_kmeans(Z, 8, 20)
    afficher_clusters(column_country, labels_Kmeans, 7)
    
    #-------------Methode CHA---------------------------------------------
    print("\nNous cherchons le nombre de clusters idéal pour la méthode CHA. \nPour cela nous utilisons la méthode du score silhouette en fonction de k")
    methode_silhouette_score(Z, 'CHA')
    
    link = linkage(Z, 'ward')
    
    plt.figure(figsize=(10,25))
    plt.title("Dendrogram ward CHA")
    plt.xlabel('distance')
    dendrogram(link, orientation = "left")
    
    print("\nVoici les clusters obtenus pour la méthode CHA Ward :\n")
    labels_CHA = fcluster(link, 26, criterion='maxclust')
    afficher_clusters(column_country, labels_CHA, 26)
    
    #-------------Methode Gaussian Mixture---------------------------------------------
    print("\nNous cherchons le nombre de clusters idéal pour la méthode mélange de Gaussiennes. \nPour cela nous utilisons la méthode du score silhouette en fonction de k")
    methode_silhouette_score(Z, 'GaussianMixture')
    
    print("\nVoici les clusters obtenus pour la méthode Gaussian Mixture :\n")
    labels_GM = apply_gaussianMixture(Z, 29, 50)
    afficher_clusters(column_country, labels_GM, 28)
    
    #-------------Methode DBSCAN---------------------------------------------
    print("\nNous cherchons les paramètres idéaux pour la méthode DBSCAN. \nPour cela nous effectuons la recherche de l'epsilon théorique.")

    print("\nNous recherchons la valeur optimal pour epsilon")
    epsilon(Z)
    eps_DBSCAN(Z, 'cluster', 15)
    eps_DBSCAN(Z, 'silhouette_score', 15)
    eps_DBSCAN(Z, 'silhouette_sample', 15)
    
    print("Nous obtenons pour DBSCAN la valeur 1.5 pour epsilon et la valeur 4 pour min_sample")
    
    print("\nVoici les clusters obtenus pour la méthode DBSCAN :\n")
    labels_DBSCAN = apply_DBSCAN(Z, 1.7, 15)
    afficher_clusters(column_country, labels_DBSCAN, 2)
    
    wait = input("****** Appuyez sur ENTREE pour passer à la partie 3.5 Clustering des données après réduction de dimension --- Partie ACP")
    
    #_________3.5 Clustering des données après réduction de dimension_________
    
    print("\n\n_______3.5 Clustering des données après réduction de dimension_______\n")

    #****************************PARTIE ACP************************************
    
    print("\n\nNous allons effectuer une ACP afin de réduire la dimension de nos données.\n")

    pca = PCA()
    pca.fit(Z)
    print("Axes / Vecteurs propres :", pca.components_)
    inertie = pca.explained_variance_ratio_
    print("\n\nVariance ratio :", inertie)
    print("\nCovariance :", pca.get_covariance())
    ValP = pca.explained_variance_
    print("\n\nValeurs propres :", ValP)
    print("\nInertie cumulée :", np.cumsum(inertie))
    print()
    
    #-------------Determiner le nombre d'axes à conserver---------------------------------------------
    
    
    #Méthode de l'éboulis des valeurs propres
    plt.subplots()
    plt.plot(np.arange(1, 10, 1),ValP)
    plt.title("Eboulis des valeurs propres")
    plt.ylabel("Valeurs propres")
    plt.xlabel("Numéro de la valeur propre")
    plt.show()
    
    #Méthode basée sur la part d'inertie cumulée
    plt.subplots()
    plt.plot(np.arange(1,10,1),np.cumsum(inertie))
    plt.title("Inertie cumulée en fonction du nombre d'axes")
    plt.ylabel("Part d'inertie cumulée")
    plt.xlabel("Nombre d'axes")
    plt.show()
     
    print("\n\n On décide, grâce à l'observation de l'éboulis des valeurs propres et de l'inertie cumulée en fonction du nombre d'axes, de conserver les 4 premiers axes.")
    sommeValP = np.sum(ValP)
    quality = (ValP[0] + ValP[1] + ValP[2] + ValP[3]) / sommeValP
    print("\nQualité globale avec 4 axes principaux:", quality)
    print()
    
    pca2 = PCA(n_components = 4)
    
    dataACP = pca2.fit_transform(Z)
    
    #Figure représentant le nuage des individus projeté dans le premier plan 
    #principal défini par les axes 1 et 2
    fig2, ax2 = plt.subplots()
    x2 = dataACP[:, 0]
    y2 = dataACP[:, 1]
    ax2.scatter(x2, y2)
    ax2.set_xlabel('axe 1', fontsize=14)
    ax2.set_ylabel('axe 2', fontsize=14)
    
    for i, txt in enumerate(column_country):
        ax2.annotate(txt, (x2[i], y2[i]), xytext=(2,8), textcoords='offset points')
        plt.scatter(x2, y2, color='green')
    
    #Figure représentant le nuage des individus projeté dans le deuxième plan 
    #principal défini par les axes 3 et 4
    fig3, ax3 = plt.subplots()
    x3 = dataACP[:, 2]
    y3 = dataACP[:, 3]
    ax3.scatter(x3, y3)
    ax3.set_xlabel('axe 3', fontsize=14)
    ax3.set_ylabel('axe 4', fontsize=14)
    
    for i, txt in enumerate(column_country):
        ax3.annotate(txt, (x3[i], y3[i]), xytext=(2,8), textcoords='offset points')
        plt.scatter(x3, y3, color='orange')
    
    #-------------Donner une interpretation des 4 axes principaux---------------------------------------------

    print("\n\n---> Nous allons à présent chercher à interpréter ces 4 axes principaux.")
    print("\nDans un premier temps nous déterminons les individus qui contribuent le plus à chacun des axes.")

    rows = dataACP.shape[0]
    cols = dataACP.shape[1]
    
    #Contribution des individus
    contrib_indiv = pd.DataFrame(np.zeros((rows, cols)))
    for i in range(0, rows):
        for j in range(0, cols):
            contrib_indiv.iloc[i, j] = (dataACP[i][j]*dataACP[i][j])/ValP[j]
    print("Contributions des individus :\n", contrib_indiv)
    print("\n\nL'individu qui contribue le plus à l'axe 1 est : ", column_country.iloc[contrib_indiv.iloc[:,0].argmax()])
    print("L'individu qui contribue le plus à l'axe 2 est : ", column_country.iloc[contrib_indiv.iloc[:,1].argmax()])
    print("L'individu qui contribue le plus à l'axe 3 est : ", column_country.iloc[contrib_indiv.iloc[:,2].argmax()])
    print("L'individu qui contribue le plus à l'axe 4 est : ", column_country.iloc[contrib_indiv.iloc[:,3].argmax()])

    #Contribution des variables

    print("\nDans un second temps nous affichons les cercles des corrélations pour les 2 premiers plans factoriels afin de déterminer la contribution des variables aux différents axes.")
    
    # Affiche le cercle des corrélations pour le premier plan factoriel
    fig_cercle1, ax_cercle1 = plt.subplots(figsize=(8, 8))
    for i in range(0, pca2.components_.shape[1]):
        ax_cercle1.arrow(0,
                 0,  # démarre une flèche à l'origine
                 pca2.components_[0, i],  #1ere composante
                 pca2.components_[1, i],  #2eme composante
                 head_width=0.1,
                 head_length=0.1)
    
        plt.text(pca2.components_[0, i] + 0.05,
                 pca2.components_[1, i] + 0.05,
                 data.columns.values[i])
    
    
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Ajoute un cercle unité sur le graphe
    plt.axis('equal')
    ax_cercle1.set_title('Cercle des corrélations dans le premier plan factoriel')
    plt.show()
    
    # Affiche le cercle des corrélations pour le deuxième plan factoriel
    fig_cercle2, ax_cercle2 = plt.subplots(figsize=(8, 8))
    for i in range(0, pca2.components_.shape[1]):
        ax_cercle2.arrow(0,
                 0,  # démarre une flèche à l'origine
                 pca2.components_[2, i],  #3eme composante
                 pca2.components_[3, i],  #4eme composante
                 head_width=0.1,
                 head_length=0.1)
    
        plt.text(pca2.components_[2, i] + 0.05,
                 pca2.components_[3, i] + 0.05,
                 data.columns.values[i])
    
    
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Ajoute un cercle unité sur le graphe
    plt.axis('equal')
    ax_cercle2.set_title('Cercle des corrélations dans le deuxième plan factoriel')
    plt.show()
    
    wait = input("****** Appuyez sur ENTREE pour passer à la partie 3.5 Clustering des données après réduction de dimension --- Partie Clustering")
  
    #****************************PARTIE CLUSTERING************************************

    
    print("\n\nMaintenant que nous avons effectué l'ACP, nous reprenons l'étude effectuée en 3.4.\n")

    print("Nous effectuons un clustering sur les nouvelles données obtenues")
    
    #-------------Methode KMeans---------------------------------------------

    print("\nNous cherchons le nombre de clusters idéal pour la méthode Kmeans. \nPour cela nous utilisons la méthode du score silhouette en fonction de k")
    methode_silhouette_score(dataACP, 'kmeans')
    
    print("\nNous obtenons 9 comme valeur optimale de k")
    
    print("\nVoici les 9 clusters obtenus pour la méthode k-means :\n")
    labels_Kmeans = apply_kmeans(dataACP, 9, 20)
    afficher_clusters(column_country, labels_Kmeans, 8)
    
    plt.figure()
    plt.scatter(dataACP[:,0],dataACP[:,1], c = labels_Kmeans)
    plt.title('ACP + Clustering K-Means')
    plt.show()

    #-------------Methode CHA---------------------------------------------
    
    print("\nNous cherchons le nombre de clusters idéal pour la méthode CHA. \nPour cela nous utilisons la méthode du score silhouette en fonction de k")
    methode_silhouette_score(dataACP, 'CHA')
    print("\nNous obtenons 28 comme valeur optimale de k")
    
    link = linkage(dataACP, 'ward')    
    print("\nVoici les 28 clusters obtenus pour la méthode CHA Ward :\n")
    labels_CHA = fcluster(link, 28, criterion='maxclust')
    afficher_clusters(column_country, labels_CHA, 28)

    plt.figure()
    plt.scatter(dataACP[:,0],dataACP[:,1], c = labels_CHA)
    plt.title('ACP + Clustering CHA')
    plt.show()
    
    #-------------Methode Gaussian Mixture---------------------------------------------
    
    print("Nous cherchons le nombre de clusters idéal pour la méthode mélange de Gaussiennes. \nPour cela nous utilisons la méthode du score silhouette en fonction de k")
    methode_silhouette_score(dataACP, 'GaussianMixture')
    print("\nNous obtenons 20 comme valeur optimale de k")
    
    print("\nVoici les 20 clusters obtenus pour la méthode Gaussian Mixture :\n")
    labels_GM = apply_gaussianMixture(dataACP, 20, 50)
    afficher_clusters(column_country, labels_GM, 19)
    
    plt.figure()
    plt.scatter(dataACP[:,0],dataACP[:,1], c = labels_GM)
    plt.title('ACP + Clustering Gaussian Mixture')
    plt.show()
    
    #-------------Methode DBSCAN---------------------------------------------
    
    print("\nNous cherchons maintenant la valeur optimale d'epsilon pour la méthode DBSCAN, nous utilisons 4 graphiques obtenus avec \n   - epsilon(dataACP)\n   - eps_DBSCAN(dataACP, 'cluster')\n   - eps_DBSCAN(dataACP, 'silhouette_score')\n   - eps_DBSCAN(dataACP, 'silhouette_sample')")
    epsilon(dataACP)
    eps_DBSCAN(dataACP, 'cluster', 24)
    eps_DBSCAN(dataACP, 'silhouette_score', 24)
    eps_DBSCAN(dataACP, 'silhouette_sample', 24)
    
    print("\nNous obtenons pour DBSCAN la valeur 0.8 pour epsilon et la valeur 4 pour min_sample")
    
   
    print("\nVoici les clusters obtenus pour la méthode DBSCAN :\n")
    labels_DBSCAN = apply_DBSCAN(dataACP, 1.4, 24)
    afficher_clusters(column_country, labels_DBSCAN, 1)
    
    plt.figure()
    plt.scatter(dataACP[:,0],dataACP[:,1], c = labels_DBSCAN)
    plt.title('ACP + Clustering DBSCAN')
    plt.show()