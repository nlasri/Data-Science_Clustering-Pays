# -*- coding: utf-8 -*-
"""Projet ODATA par Narjisse LASRI et Estelle DUHEM"""

from part2 import part2
from part3 import part3

user_input = input("--->Entrez 2 pour lancer la partie 2 du projet\n--->Entrez 3 pour lancer la partie 3 du projet\n")
if user_input == '2':
    part2()
elif user_input == '3':
    part3()

