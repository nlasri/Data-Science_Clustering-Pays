# -*- coding: utf-8 -*-
"""Projet ODATA par Narjisse LASRI et Estelle DUHEM"""

import pandas as pd
import numpy as np
from sklearn import metrics


from functions import *

def part2():
    #_________2.3 Expérimentations_________
    
    #--> Importation des données
    
    
    g2_20 = pd.read_csv('g2-2-20.txt', sep = '     ', engine = 'python', names=['column1', 'columns2'])
    g2_100 = pd.read_csv('g2-2-100.txt', sep = '     ', engine = 'python', names=['column1', 'columns2'])
    jain = pd.read_csv('jain.txt', sep = '	', engine = 'python', names=['column1', 'columns2', 'column3'])
    aggregation = pd.read_csv('Aggregation.txt', sep = '	', engine = 'python', names=['column1', 'columns2', 'column3'])
    pathbased = pd.read_csv('pathbased.txt', sep = '	', engine = 'python', names=['column1', 'columns2', 'column3'])
    
    
    #--> Visualisation des nuages de points
    
    
    affichage_plot('Nuage de points de g2-2-20', g2_20)
    affichage_plot('Nuage de points de g2-2-100', g2_100)
    affichage_plot('Nuage de points de jain', jain, jain.iloc[:,2].to_numpy())
    affichage_plot('Nuage de points de aggregation', aggregation, aggregation.iloc[:,2].to_numpy())
    affichage_plot('Nuage de points de pathbased', pathbased, pathbased.iloc[:,2].to_numpy())
    
    
    #--> Méthode k-means 
    
    
    
    affichage_plot('Résultat k-means sur g2-2-20', g2_20, apply_kmeans(g2_20, 2, 4))
    affichage_plot('Résultat k-means sur g2-2-100', g2_100, apply_kmeans(g2_100, 4, 10))
    affichage_plot('Résultat k-means sur jain', jain, apply_kmeans(jain, 2, 10))
    affichage_plot('Résultat k-means sur aggregation', aggregation, apply_kmeans(aggregation, 7, 20))
    affichage_plot('Résultat k-means sur pathbased', pathbased, apply_kmeans(pathbased, 3, 5))
    
    
    #--> Méthode CHA
    
    
    affichage_plot('Résultat CHA Complete sur g2-2-20', g2_20, apply_CHA(g2_20, 'complete', 2, True))
    affichage_plot('Résultat CHA Complete sur g2-2-100', g2_100, apply_CHA(g2_100, 'complete', 4, True))
    affichage_plot('Résultat CHA Single sur jain', jain, apply_CHA(jain, 'single', 2, True))
    affichage_plot('Résultat CHA Complete sur aggregation', aggregation, apply_CHA(aggregation, 'complete', 7, True))
    affichage_plot('Résultat CHA Ward sur pathbased', pathbased, apply_CHA(pathbased, 'ward', 3, True))
    
    
    #--> Méthode mélange de Gaussiennes
    
    
    
    affichage_plot('Résultat Gaussian Mixture de g2-2-20', g2_20, apply_gaussianMixture(g2_20, 2))
    affichage_plot('Résultat Gaussian Mixture sur g2-2-100', g2_100, apply_gaussianMixture(g2_100, 4))
    affichage_plot('Résultat Gaussian Mixture sur jain', jain, apply_gaussianMixture(jain, 2))
    affichage_plot('Résultat Gaussian Mixture sur aggregation', aggregation, apply_gaussianMixture(aggregation, 7, 3))
    affichage_plot('Résultat Gaussian Mixture sur pathbased', pathbased, apply_gaussianMixture(pathbased, 3))
    
    
    #--> Méthode DBSCAN
    
    
    epsilon(g2_20)
    epsilon(g2_100)
    epsilon(jain)
    epsilon(aggregation)
    epsilon(pathbased)
    
    
    
    affichage_plot('Résultat DBSCAN de g2-2-20', g2_20, apply_DBSCAN(g2_20, 10, 5))
    affichage_plot('Résultat DBSCAN de g2-2-100', g2_100, apply_DBSCAN(g2_100, 30, 10))
    affichage_plot('Résultat DBSCAN de jain', jain, apply_DBSCAN(jain, 2.7, 3))
    affichage_plot('Résultat DBSCAN de aggregation', aggregation, apply_DBSCAN(aggregation, 1.5 , 5))
    affichage_plot('Résultat DBSCAN de pathbased', pathbased, apply_DBSCAN(pathbased, 2 , 10))
    
    
    #Tableau des scores
    
    score = np.empty([4, 3]) # tableau qui contient les scores de chaque échantillon avec les différentes méthodes
    
    #méthode Kmeans
    score[0][0]= metrics.adjusted_rand_score(apply_kmeans(jain, 2, 10), jain.iloc[:,2].to_numpy())
    score[0][1]= metrics.adjusted_rand_score(apply_kmeans(aggregation, 7, 20), aggregation.iloc[:,2].to_numpy())
    score[0][2]= metrics.adjusted_rand_score(apply_kmeans(pathbased, 3, 5), pathbased.iloc[:,2].to_numpy())
    
    #méthode CHA
    score[1][0]= metrics.adjusted_rand_score(apply_CHA(jain, 'single', 2), jain.iloc[:,2].to_numpy())
    score[1][1]= metrics.adjusted_rand_score(apply_CHA(aggregation, 'complete', 7), aggregation.iloc[:,2].to_numpy())
    score[1][2]= metrics.adjusted_rand_score(apply_CHA(pathbased, 'ward', 3), pathbased.iloc[:,2].to_numpy())
    
    #méthode mélange de Gaussiennes
    score[2][0]= metrics.adjusted_rand_score(apply_gaussianMixture(jain, 2), jain.iloc[:,2].to_numpy())
    score[2][1]= metrics.adjusted_rand_score(apply_gaussianMixture(aggregation, 7, 3), aggregation.iloc[:,2].to_numpy())
    score[2][2]= metrics.adjusted_rand_score(apply_gaussianMixture(pathbased, 3), pathbased.iloc[:,2].to_numpy())
    
    #méthode DBSCAN
    score[3][0]= metrics.adjusted_rand_score(apply_DBSCAN(jain, 2.7, 3), jain.iloc[:,2].to_numpy())
    score[3][1]= metrics.adjusted_rand_score(apply_DBSCAN(aggregation, 1.5 , 5), aggregation.iloc[:,2].to_numpy())
    score[3][2]= metrics.adjusted_rand_score(apply_DBSCAN(pathbased, 2, 5), pathbased.iloc[:,2].to_numpy())
    
    #On crée un tableau avec des noms de colonnes et de lignes pour score pour permettre une meilleure lecture
    ARI_scores = pd.DataFrame(score, index = ['KMeans', 'CHA', 'Gaussian Mixture', 'DBSCAN'], columns = ['jain','aggregation','pathbased'])
    print("\nIndices de Rand Ajustés : \n", ARI_scores)

