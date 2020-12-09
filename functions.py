# -*- coding: utf-8 -*-
"""Projet ODATA par Narjisse LASRI et Estelle DUHEM"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import silhouette_score, silhouette_samples

def affichage_plot(plot_title, dataset, couleur=np.array([])):
    """
        Cette fonction permet d'afficher sur une nouvelle figure le nuage
        de points d'un ensemble de données, et de colorer ces points. 
 
        :param plot_title: Titre du graphique (type : chaîne de caractères)
        :param dataset: Ensemble de données à représenter (type : Pandas DataFrame)
        :param couleur: Données permettant de colorer les points (type : Numpy Array)
    
    """
    plt.figure()
    if (couleur.size == 0):
        plt.scatter(dataset.iloc[:,0],dataset.iloc[:,1])
    else:
        plt.scatter(dataset.iloc[:,0],dataset.iloc[:,1], c = couleur)
    plt.title(plot_title)
    plt.show()



def apply_kmeans(data, nbClusters, nbInit):
    """
        Cette fonction permet d'appliquer la méthode KMeans de sklearn.cluster
        à un ensemble de données en choisissant le nombre de clusters souhaités
        et le nombre d'initialisations différentes à effectuer pour les centres
        initiaux des clusters.
 
        :param data: Ensemble de données (type : Pandas Dataframe)
        :param nbClusters: Nombre de clusters souhaités (type : int)
        :param nbInit: Nombre d'initialisations à effectuer (type : int)
    
    """
    kmean = KMeans(n_clusters=nbClusters, n_init = nbInit).fit(data)
    kmean.transform(data)
    labels = kmean.labels_
    return labels



def plot_dendrogram(Z, names):
    """
        Cette fonction permet d'afficher sur une nouvelle figure le 
        dendrogramme obtenu avec la matrice linkage du clustering
        hiérarchique.
 
        :param Z: Matrice linkage dont il faut représenter le dendrogramme
        :param names: Noms des données
    
    """
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()



def apply_CHA(data, methode, max_clust, boolean=False) :
    """
        Cette fonction permet d'appliquer la méthode CHA (Clustering hiérarchique 
        ascendant) de scipy.cluster.hierarchy à un ensemble de données en 
        choisissant la méthode, la métrique, le nombre de clusters et en 
        choissisant si l'on souhaite afficher le dendrogramme associé ou non.
 
        :param data: Ensemble de données (type : Pandas Dataframe)
        :param methode: Paramètre 'method' de la fonction linkage()
        :param max_clust: Nombre de clusters souhaités (type : int)
        :param boolean: Si True on affiche le dendrogramme, si False non
    
    """
    # préparation des données pour le clustering
    X = data.values
    names = data.index

    # Centrage et Réduction
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)

    # Clustering hiérarchique
    Z = linkage(X_scaled, method= methode)
    
    if boolean == True :
        # Affichage du dendrogramme
        plot_dendrogram(Z, names)
    
    #Construction des clusters
    clusters = fcluster(Z, max_clust, criterion='maxclust')
    return clusters
    

def apply_gaussianMixture(data, nb_components, nb_init=1):
    
    """
        Cette fonction permet d'appliquer la méthode Mélange de Gaussiennes 
        de sklearn.mixture à un ensemble de données en 
        choisissant le nombre de clusters et le nombre d'initialisations
 
        :param data: Ensemble de données (type : Pandas Dataframe)
        :param nb_components: Nombre de clusters 
        :param nb_init: Nombre d'initialisations (par défaut à 1)
    
    """
    
    gmm = GMM(n_components = nb_components, n_init = nb_init).fit(data)
    return gmm.predict(data)



def epsilon(data):
    
    """
        Cette fonction permet de rechercher le paramètre epsilon de la méthode DBSCAN
        
        :param data: Ensemble de données (type : Pandas Dataframe)
     
    """
    
    voisins = NearestNeighbors(n_neighbors=2)
    voisinage = voisins.fit(data)
    distances, indices = voisinage.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.figure()
    plt.plot(distances)
    plt.ylabel('eps')
    plt.title("Choix d'espsilon pour DBSCAN")
    plt.show()


def apply_DBSCAN(data, epsilon, minimum_samples):
    """
        Cette fonction permet d'appliquer la méthode DBSCAN de sklearn.cluster
        à un ensemble de données en choisissant les paramètres epsilon et min_samples
 
        :param data: Ensemble de données (type : Pandas Dataframe)
        :param epsilon: distance maximale entre deux points pour que l’un soit considéré comme le voisinage de l’autre
        :param minimum_samples: Nombre minimal de points dans le voisinage d’un point central.

    """
    clustering = DBSCAN(eps= epsilon, min_samples = minimum_samples).fit(data)
    return clustering.labels_


def afficher_histogrammes(data):
    
    """
        Cette fonction permet d'afficher les histogrammes des attributs des données
        
        :param data: Ensemble de données (type : Pandas Dataframe)
     
    """
    
    figA,axA = plt.subplots(3)
    axA[0].hist(data['child_mortality'],color = "pink", label = "child_mortality", histtype='stepfilled', bins=120)
    axA[0].legend()
    axA[1].hist(data['exports'],color = "grey", label = "exports", histtype='stepfilled', bins=120)
    axA[1].legend()
    axA[2].hist(data['health'],color = "purple", label = "health", histtype='stepfilled', bins=120)
    axA[2].legend()
    figB,axB = plt.subplots(3)
    axB[0].hist(data['imports'],color = "pink", label = "imports", histtype='stepfilled', bins=120)
    axB[0].legend()
    axB[1].hist(data['income'],color = "grey", label = "income", histtype='stepfilled', bins=120)
    axB[1].legend()
    axB[2].hist(data['inflation'],color = "purple", label = "inflation", histtype='stepfilled', bins=120)
    axB[2].legend()
    figC,axC = plt.subplots(3)
    axC[0].hist(data['life_expectation'],color = "pink", label = "life_expectation", histtype='stepfilled', bins=120)
    axC[0].legend()
    axC[1].hist(data['total_fertility'],color = "grey", label = "total_fertility", histtype='stepfilled', bins=120)
    axC[1].legend()
    axC[2].hist(data['GDP'],color = "purple", label = "GDP", histtype='stepfilled', bins=120)
    axC[2].legend()
    
    
def remplacer_val_aberrantes(data):
    

    """
        Cette fonction de détecter les données aberrantes avec le critère des 
        trois sigmas sur les attributs qui semblent suivre une loi normale et 
        qui remplace ces données par NaN
        
        :param data: Ensemble de données (type : Pandas Dataframe)
     
    """
    
    
    j=0
    temp = data.drop('country', 1)
    for column,value in temp.iteritems(): 
        m = data[str(column)].mean()
        e = data[str(column)].std()
        for i in value:
            if (i>m+3*e or i<m-3*e) & (str(column)=='exports' or str(column)=='health' or str(column)=='imports' ):
                print("\nLa valeur : ", i, "de la colonne : ", str(column),"et du pays : ", data.iloc[data[data[str(column)]==i].index.values,0] ,"est aberrante" )
                temp[temp==i]= np.nan
                j=j+1
    print ('\nNombre de données aberrantes : ' + str(j))
    return temp, data['country']

def remplacer_val_manquantes(matrice, choix):
    
    """
        Cette fonction remplace les valeurs nulles d'une colonne par la moyenne 
        ou la médiane de celle-ci
 
        :param data: Ensemble de données (type : Pandas Dataframe)
        :param choix: Choix de la moyenne ou de la medianne

    """
    for column in matrice :
        if (choix == "moyenne"):
            remplacement = matrice[column].mean()  #calculer moyenne
        elif (choix == "mediane"):
            remplacement = matrice[column].median() #calculer median
        matrice[column] = matrice[column].fillna(remplacement) #remplacer les valeurs nulles par la moyenne des valeurs
    return matrice


def liste_corr(dataset, threshold):
    
    """
        Cette fonction renvoie la liste de tous les attributs corrélés
 
        :param dataset: Ensemble de données (type : Pandas Dataframe)
        :param threshold: seuil à partir duquel 2 attributs sont corrélés

    """

    
    list_attributes_corr = list() #Liste des couples d'attributs fortement corrélés
    corr_matrix = dataset.corr(method='spearman') 
    for i in range(len(corr_matrix.columns)):
        for j in range(i): # on parcourt le triangle supérieur de la matrice
            if (abs(corr_matrix.iloc[i, j]) >= threshold):
                attributes_corr = corr_matrix.columns[i] +" et "+corr_matrix.columns[j] # On prend les attributs corrélés
                list_attributes_corr.append(attributes_corr)
    return list_attributes_corr



def liste_corr_max(dataset, choix):
    
    
    """
        Cette fonction renvoie la liste des attributs les plus corréles, 
        positivement, négativement, positivement et négativement ou les 
        attributs les moins corrélés 
 
        :param dataset: Ensemble de données (type : Pandas Dataframe)
        :param choix: choix de la corrélation ("positif", "negatif", 
        "max" (attributs les plus corrélés) ou "autre" (attributs les moins 
        corrélés))

    """
    
    
    max = -1
    max1 = 1
    attributes_corr = None
    corr_matrix = dataset.corr(method='spearman') #matrice de corrélation 
    for i in range(len(corr_matrix.columns)):
        for j in range(i): # on parcourt le triangle supérieur de la matrice
            if (choix == "positif") or (choix == "max"):
                if (choix == "max"):
                    corr = abs(corr_matrix.iloc[i, j])
                else:
                    corr = corr_matrix.iloc[i, j]
                    
                if (corr >= max):
                    max = corr_matrix.iloc[i, j];
                    attributes_corr = corr_matrix.columns[i] +" et "+corr_matrix.columns[j] # On prend les attributs les plus corrélés
            else:
                if (choix == "autre"):
                    corr = abs(corr_matrix.iloc[i, j])
                else:
                    corr = corr_matrix.iloc[i, j]
                    
                if (corr <= max1):
                    max1 = corr;
                    max = corr_matrix.iloc[i, j]
                    attributes_corr = corr_matrix.columns[i] +" et "+corr_matrix.columns[j]
    return attributes_corr, max
    
def elbow_method_kmeans(data):
        
    """
        Cette fonction trace un graphique de l’inertie des données avec la 
        méthode K-means en fonction du nombre de clusters
 
        :param dataset: Ensemble de données (type : Pandas Dataframe)

    """
    distortions = []
    K = range(1,20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    
def methode_silhouette_score(data, choix):
    
    """
        Cette fonction trace un graphique du score silhouette avec la méthode 
        choisie (K-means, CHA, Mélange de Gaussiennes) en fonction du nombre de clusters
 
        :param dataset: Ensemble de données (type : Pandas Dataframe)
        :param choix: Méthode choisie ('kmeans', 'CHA', 'GaussianMixture')

    """
    
    scores = []
    K = range(2, 40)
    for k in K:
        if(choix == 'kmeans'):
            kmean = KMeans(n_clusters = k,init="k-means++",n_init = 20).fit(data)
            kmean.transform(data)
            scores.append(silhouette_score(data, kmean.labels_))
            
        elif(choix == 'CHA'):
            link = linkage(data, 'ward')
            labels = fcluster(link, k, criterion='maxclust')
            scores.append(silhouette_score(data, labels))
        else:
            labels = apply_gaussianMixture(data, k, 20)
            scores.append(silhouette_score(data, labels))
        
        
    plt.figure(figsize=(16,8))
    plt.plot(K, scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    
    if (choix == 'kmeans'):
        plt.title('Silhouette Method for k means')
     
    elif (choix == 'CHA'):
        plt.title('Silhouette Method for CHA')
           
    else:
        plt.title('Silhouette Method for Gaussian Mixture')

    plt.show()
    
def eps_DBSCAN(data, choix, nb_samples):
    
    """
        Cette fonction aide à trouver epsilon, un des paramètres de DBSCAN.
    
        3 méthodes sont possibles :
        silhouette_sample : évolution du nombre d'échantillons dans le 
        bon cluster en fonction de l'évolution de epsilon
        
        silhouette_score : évolution du score silhouette en fonction de epsilon
        
        cluster : évolution du nombre de clusters en fonction de epsilon
 
        :param dataset: Ensemble de données (type : Pandas Dataframe)
        :param choix: Méthode choisie ('silhouette_sample', 'silhouette_score', 'cluster')

    """

    
    scores = []
    EPSILON = np.arange(0.6, 2.6, 0.1)
    for epsilon in EPSILON:
        dbscan = apply_DBSCAN(data, epsilon, nb_samples)
        
        if (choix == 'silhouette_sample'):
            resultat = 0
            if (dbscan.max() > 0):
                for i in silhouette_samples(data, dbscan):
                    if (i > 0):
                        resultat = resultat + 1
            scores.append(resultat)    
        
        elif(choix == 'silhouette_score'):
            if (dbscan.max() > 0):
                scores.append(silhouette_score(data, dbscan))
            else:
                scores.append(0)

        elif (choix == 'cluster'):
            scores.append(len(set(dbscan))-1)
        
    plt.figure(figsize=(16,8))
    plt.plot(EPSILON, scores, 'bx-')
    plt.xlabel('eps')
    plt.ylabel('result')
    if (choix == 'silhouette_sample'):
        plt.title("Evolution du nombre d'échantillons dans le bon cluster en fonction de l'évolution de epsilon")
        
    elif(choix == 'silhouette_score'):
        plt.title("Evolution du score silhouette en fonction de epsilon")
        
    elif (choix == 'cluster'):
        plt.title("Evolution du nombre de clusters en fonction de epsilon")
    
    plt.show()
    
def afficher_clusters(column_country, labels, clusters):
    
    """
        Cette fonction permet d'afficher les clusters
     
        :param dataset: Ensemble de données (type : Pandas Dataframe)

    """
    
    
    for i in range( -1, clusters + 1):
        j=0
        classe = []
        for value in labels:
            if (value == i):
                classe.append(column_country[j])
            j = j + 1
        print("\nlist", i, " : \n", classe)

    

