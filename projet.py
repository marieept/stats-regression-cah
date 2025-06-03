import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import math
import pandas as pd

points =[(1,1),(1,2),(1,5),(3,4),(4,3),(6,2),(0,4)]
noms= ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]

#########################################
#               Partie 1                #
#########################################

print("\nPARTIE 1\n")

#Séparer les coordonnées x et y
x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])


def afficher_stats(valeurs, noms):
    print("Moyenne :", np.mean(valeurs))
    print("Médiane :", np.median(valeurs))
    print("Variance :", np.var(valeurs))
    print("Ecart-type :", np.std(valeurs))
    print("Minimum :", np.min(valeurs))
    print("Maximum :", np.max(valeurs))
    print("Etendue :", np.ptp(valeurs))


afficher_stats(x, "x")
afficher_stats(y, "y")

plt.scatter(x, y, color='blue')
for i in range(len(points)):
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i])
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

#   Interprétation statistique :

#Les valeurs xx sont plus dispersées que les valeurs yy.
#La moyenne de xx est plus faible que celle de yy, indiquant une asymétrie spatiale.
#L’étendue plus grande pour xx traduit une plus grande variabilité sur l’axe horizontal.

#   Interprétation graphique :

#Le nuage montre une certaine tendance croissante mais non strictement linéaire.
#Il est possible qu’une relation linéaire approximative existe entre xx et yy, à tester dans la partie suivante.


#########################################
#               Partie 2                #
#########################################

print("\nPARTIE 2\n")

# 2.1 Calcul des Coefficients de Régression

# Moyennes
x_moy = np.mean(x)
y_moy = np.mean(y)

# Coefficients
b1 = np.sum((x - x_moy) * (y - y_moy)) / np.sum((x - x_moy)**2)
b0 = y_moy - b1 * x_moy

print("bo =", b0)
print("b1 =", b1)

# Modèle de prédictions
y_pred = b0 + b1 * x


# 2.3 Coefficient de Détermination R²

# R²
SCE = np.sum((y - y_pred) ** 2)
SCT = np.sum((y - y_moy) ** 2)
SCR = np.sum((y_pred - y_moy) ** 2)
R2 = 1 - SCE / SCT
print("R² (Coefficient de détermination) :", R2)


# 2.2 Visualisation de la Droite de Régression

# Affichage
plt.scatter(x, y, label='Points')
plt.plot(x, y_pred, color='red', label=f'Droite de régression: y = {b0:.2f} + {b1:.2f}x')
plt.legend()
plt.title(f'Droite de régression linéaire simple (R² = {R2:.2f})')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
plt.savefig("figure_2_1.jpg")
print("Graphique enregistré dans le fichier figure_2_1.jpg")


#########################################
#               Partie 3                #
#########################################

print("\nPARTIE 3\n")

#1. Résidus et somme des carrés des erreurs (SCE)
e = y - y_pred

SCE=0

for i in range(len(points)):
    SCE += e[i]**2 
print("SCE (Somme des carrés des erreurs) :", SCE)

#2. Etimation de la variance des erreurs : MSE
n= len(x)

MSE = SCE / (n-2)

print("MSE (Erreur quadratique moyenne) :", MSE)

#Le n-2 vient du fait qu'on a estimé 2 paramètres (b0 et b1). On parle alors de degrés de liberté.

#3. Ecart-type des erreurs
s = np.sqrt(MSE)

print("Écart-type des erreurs :", s)

#4. Interprétation des Résultats

# 🔹 Interprétation des coefficients

#     b0=3.33 (ordonnée à l'origine) : c'est la valeur estimée de y quand x=0. Cela signifie qu'à l'origine de l'axe x, la droite de régression prévoit y=3.33.

#     b1=−0.15 (pente) : chaque augmentation de 1 unité en x entraîne une baisse moyenne de y de 0.15. La relation est donc légèrement décroissante, mais très faible.

# 🔹 Coefficient de détermination R2

#     Le R2=0.049 (soit 4.9%) indique que seulement 4.9% de la variation de y est expliquée par la variable x.

#     Cela signifie que la droite de régression explique très peu la variabilité des points. La majorité de la variation de y provient donc d'autres facteurs non capturés par ce modèle.

# 🔹 Analyse des erreurs

#     SCE (Somme des carrés des erreurs) : 11.42
#     → mesure l’erreur globale du modèle (plus elle est faible, meilleur est l’ajustement).

#     MSE (Erreur quadratique moyenne) : 2.28
#     → estimation de la variance des erreurs résiduelles.

#     Écart-type des erreurs : 1.51
#     → en moyenne, les prédictions du modèle s’écartent de 1.51 unités des valeurs réelles.

#     Comparé à l’écart-type total de y qui est de 1.31, cela montre que la droite n'améliore pas vraiment la prédiction par rapport à une moyenne constante.

#########################################
#               Partie 4                #
#########################################

print("\nPARTIE 4\n")

# 4.1 Estimation de la variance des erreurs

# 4.2 Erreurs Standards des Coefficients

# Erreur standard de la pente
somme_x=0
for i in range(len(points)):
    somme_x += (x[i]-x_moy)**2

SEb1 = s/np.sqrt(somme_x)
print("Erreur standard de la pente (SEb1):", SEb1)

# Erreur standard de l'ordonnée à l'origine
SEb0 = s*np.sqrt(1/n+x_moy**2/somme_x)
print("Erreur standard de l'ordonnée à l'origine (SEb0)", SEb0)
print("\n")

# 4.3 Test d'Hypothèse pour la Pente (b1)

# Hypothèses :
# H0 : b1 = 0 (pas de relation linéaire)
# H1 : b1 != 0 (relation linéaire significative)

# Statistique de test

t_b1 = (b1-0)/SEb1 # t suit une loi de Student donc on calcul la p-valeur pour savoir si on rejette l'hypothèse ou non
p_b1 = 2 * (1 - stats.t.cdf(abs(t_b1), df=n - 2))
print("b1 :")
print("valeur de test", t_b1)
print("p-valeur:", p_b1)
if(p_b1<0.05):
    print("p-valeur<0,05 donc on rejette l'hypothèse, b1 est significatif")
else:
    print("p-valeur>0,05 donc on ne rejette pas l'hypothèse, il n'y a pas de preuve de relation linéaire")
print("b1 = ", b1)
print("\n")

# 4.4 Test d'Hypothèse pour l'Ordonnée à l'Origine (b0)
# Hypothèses : H0 : b0 = 0 vs H1 : b0 != 0

# Statistique de test

t_b0 = (b0-0)/SEb0
p_b0 = 2 * (1 - stats.t.cdf(abs(t_b0), df=n - 2))
print("b0 :")
print("valeur de test", t_b0)
print("p-valeur:", p_b0)
if(p_b0<0.05):
    print("p-valeur<0,05 donc on rejette l'hypothèse, b0 est significatif, la constante n'est pas nulle")
else:
    print("p-valeur>0,05 donc on ne rejette pas l'hypothèse, il n'y a pas de preuve de relation linéaire, b0 n'est pas significatif")
print("b0 = ", b0)
print("\n")

# 4.5 Intervalles de Confiance pour les Coefficients

alpha = 0.05

t_crit = stats.t.ppf(1-alpha/2, df= n-2)

IC_b1=[float(b1-t_crit*SEb1),float(b1+t_crit*SEb1)]
IC_b0=[float(b0-t_crit*SEb0),float(b0+t_crit*SEb0)]

print("Intervalle de confiance à 95% pour b1:", IC_b1)
print("Intervalle de confiance à 95% pour b0:", IC_b0)

# 4.6 Interprétation des Tests Statistiques

# Pour l'erreur standard des coefficients on a :
#   Erreur standard de la pente SEb1 = 0.2885
#   Erreur standard à l'ordonnée à l'origine SEb0 = 0.8724

# Test d'hypothèse pour la pente (b1) :
#   Hypothèses :
#        H0 : b1 = 0 (pas de relation linéaire)
#        H1 : b1 != 0 (relation linéaire significative)
#   Statistique de test t : -0.5054
#   p-valeur : 0.6347 (> 0.05)
#   Conclusion :
#       On ne rejette pas H0.
#       Il n’y a pas de preuve statistique d’une relation linéaire significative entre les variables.
#       Autrement dit, la pente n’est pas significative.

# Test d'hypothèse pour l'ordonnée à l'origine (b0) :
#   Hypothèses :
#        H0 : b0 = 0 (la constante est nulle)
#        H1 : b0 != 0 (la constante est significativement différente de 0)
#   Statistique de test t : 3.8208
#   p-valeur : 0.0124 (< 0.05)
#   Conclusion :
#       On rejette H0.
#       La constante est significativement différente de 0.

# Intervalle de confiance à 95% :
#   Pour b1 (la pente):
#       [-0.8875, 0.5958]
#       Contient zéro donc cohérent avec le fait que b1 n’est pas significatif.
#   Pour b0 (l'ordonnée à l'origine) :
#       [1.0907, 5.5760]
#       Ne contient pas zéro donc confirme que b0 est significatif.

# Interprétation finale : 
#   La constante b0 est significative : le modèle prédit une valeur moyenne de y autour de 3.33 même quand x = 0.
#   La pente b1 n’est pas significative : la variable explicative x n’explique pas de manière significative la variation de y dans notre échantillon.
#   On pourrait envisager d'autres modèles, transformations de variables, ou d’augmenter la taille de l’échantillon si on soupçonne une relation qui n’apparaît pas ici.

#########################################
#               Partie 5                #
#########################################

print("\nPARTIE 5\n")

# Classification Ascendante Hiérarchique (CAH)

# 1.a
def dist(p1, p2):
    '''
    Cette fonction prend deux couples de points en paramètre et retourne une distance entre ces deux points
    :param p1 : premier couple de point
    :param p2 : second couple de point
    :return : la distance entre ces deux points
    '''
    """Distance euclidienne"""
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    print("La distance euclidienne vaut : ", distance)
    return distance

# 1.b
def dist1(p1, p2):
    '''
    Cette fonction prend deux couples de points en paramètre et retourne une distance entre ces deux points
    :param p1 : premier couple de point
    :param p2 : second couple de point
    :return : la distance entre ces deux points
    '''
    """Distance de Manhattan"""
    distance = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    print("La distance de Manhattan vaut : ", distance)
    return distance

# 1.c
def dist_inf(p1, p2):
    '''
    Cette fonction prend deux couples de points en paramètre et retourne une distance entre ces deux points
    :param p1 : premier couple de point
    :param p2 : second couple de point
    :return : la distance entre ces deux points
    '''
    """Distance de Chebyshev (max)"""
    distance = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    print("La distance de Chebyshev vaut : ", distance)
    return distance

# 1.d : Distance de Ward ???


# 2.
def dist_min(tableau, dist_func):
    '''
    Cette fonction prend un tableau et une fonction en paramètre et retourne un couple de point situés à une distance minimale l'un de l'autre
    :param : tableau : un tableau de points
    :param : dist_func : la fonction du calcul de la distance entre deux points que l'on veut utiliser
    :return : un couple de points
    '''
    min_d = float('inf')
    couple_points = None
    for i in range(len(tableau)):
        for j in range(i+1, len(tableau)):                  #Permet de parcourir toutes les lignes et toutes les colonnes d'un tableau 
            d = dist_func(tableau[i], tableau[j])           #Récupère la distance à partir de la fonction donnée en entrée
            if d < min_d:                                   #Calcul de toutes les distances entre les points et garde la distance minimum
                min_d = d
                couple_points = (tableau[i], tableau[j])
    return couple_points, min_d


# 3.

# Initialisation de la matrice
n = len(x)
matrice_1 = np.zeros((n, n))

# Remplissage de la matrice avec d²
for i in range(n):
    for j in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        d_eucli = dist(points[i], points[j])
        d_eucli_2 = d_eucli**2
        matrice_1[i][j] = d_eucli_2

# Affichage avec pandas pour lisibilité
data = pd.DataFrame(matrice_1, index=y, columns=x)
print("Matrice des distances euclidiennes au carré :\n")
print(data.round(1))

# Tracé
plt.scatter(x, y, color='blue')
for i in range(len(points)):
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i])
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")

pair_min, d_min = dist_min(points, dist)

print("La paire de points les plus proche est : ", pair_min)

x_vals = [pair_min[0][0], pair_min[0][1]]
y_vals = [pair_min[1][0], pair_min[1][1]]

#x_vals, y_vals = dist_min(matrice_1, dist)

# Encadrer les 2 points les plus proches (Classe Γ₁)
plt.plot(x_vals, y_vals, 'go--', label="Classe Γ1")
plt.scatter(x_vals, y_vals, color='green')
plt.title("Regroupement en classes")
plt.grid(True)
plt.xlim(-1, 7)
plt.ylim(0, 6)

# plt.savefig("figure_5_3.jpg")
# print("Graphique enregistré dans le fichier figure_5_3.jpg")


#4.

classe_G1= list(pair_min)
print("Points dans Γ2 :", classe_G1)

points_restants=[p for p in points if p not in classe_G1]

n_total = len(points_restants)

matrice_2 = np.zeros((n_total+1, n_total+1))

def dist_groupe_point(groupe, point):
    return min(dist(g, point) for g in groupe)

matrice_2[0][0]=0

#Distances entre Γ1 et les points restants
for i, p in enumerate(points_restants):
    d= dist_groupe_point(classe_G1, p)
    matrice_2[0, i+1]=d
    matrice_2[i+1, 0]=d
    
#Distances entre les points restants
for i in range(n_total):
    for j in range(n_total):
        if i == j:
            matrice_2[i+1, j+1] = 0
        else:
            matrice_2[i+1][j+1]= dist(points_restants[i], points_restants[j])

noms_groupes = ["Γ1"] + [f"{noms[points.index(p)]}" for p in points_restants]

df2 = pd.DataFrame(matrice_2, index=noms_groupes, columns=noms_groupes)
print("Matrice des distances euclidiennes au carré :\n")
print(df2.round(1))


#GAMMA 2

# Trouver la paire avec la distance minimale (hors diagonale 0)
min_dist = float('inf') #on initialise à l'infini
fusion_indices = (None, None)

n = matrice_2.shape[0]

#Double boucle pour trouver la distance minimale au dessus de la diagonale
for i in range(n):
    for j in range(i+1, n):
        if matrice_2[i, j] < min_dist:
            min_dist = matrice_2[i][j]
            fusion_indices = (i, j)

# On construit la liste des points formant Γ2 (fusion entre Γ1 et un point ou entre 2 points isolés)
if 0 in fusion_indices:
    autre_idx = fusion_indices[1] if fusion_indices[0] == 0 else fusion_indices[0]
    classe_G2 = classe_G1 + [points_restants[autre_idx - 1]]
else:
    classe_G2 = [points_restants[fusion_indices[0] - 1], points_restants[fusion_indices[1] - 1]]

print("Points dans Γ2 :", classe_G2)


# On récupère les indices des deux groupes à fusionner
i1, i2 = fusion_indices

# Si l'un est Γ1 (indice 0), on récupère le point isolé fusionné
if i1 == 0:
    p1 = classe_G1[0]
    p2 = points_restants[i2 - 1] #-1 car points_restants ne contient pas G1
elif i2 == 0:
    p1 = classe_G1[0]
    p2 = points_restants[i1 - 1]
else:
    # Fusion entre deux points isolés
    p1 = points_restants[i1 - 1]
    p2 = points_restants[i2 - 1]

# Tracer la ligne entre les deux points formant Γ2
x_vals2= [p1[0], p2[0]]
y_vals2=[p1[1], p2[1]]

plt.plot(x_vals2, y_vals2, 'ro--', label="Classe Γ2")
plt.scatter(x_vals2, y_vals2, color='red')

plt.legend()
plt.show()

#GAMMA 3



# 6.

# Étape 1 : calcul de la matrice de linkage avec méthode "single"
# La fonction linkage construit la hiérarchie de regroupement en utilisant la distance euclidienne.
# Ici, on utilise la méthode 'single' (plus proche voisin), ce qui signifie que la distance entre deux groupes est définie comme la distance minimale entre un point de l'un et un point de l'autre.
# Cette ligne automatise toutes les étapes de fusion des groupes jusqu'à ce qu'il n'en reste plus qu'un.
linked = linkage(points, method='single', metric='euclidean') # La variable linked contient toutes les étapes de regroupement.

# Étape 2 : Affichage du dendrogramme
# La fonction dendrogram() affiche visuellement les fusions successives entre groupes.
# Chaque branche représente une étape de regroupement ; plus elle est haute, plus la distance entre les groupes fusionnés est grande.
# Les labels permettent de visualiser quels points (M1 à M7) sont fusionnés à chaque étape.
# Cela remplace et résume graphiquement toutes les étapes manuelles de la classification (Γ1, Γ2, etc.).
plt.figure(figsize=(10, 6))
dendrogram(linked, labels=noms)
plt.title('Dendrogramme - CAH (single linkage)')
plt.xlabel('Points')
plt.ylabel('Distance euclidienne')
plt.grid(True)
plt.show()

# 7.
# On repart du dendogramme que l'on a construit dans la question 6 
plt.figure(figsize=(10, 6))
dendrogram(linked, labels=noms)
plt.title('Dendrogramme - CAH (single linkage)')
plt.xlabel('Points')
plt.ylabel('Distance euclidienne')
seuil = 2 # seuile de découpe, on ne peut pas mettre au-dessus de la hauteur finale du dendogramme
# le seuil de découpe permet de former des groupes, il faut couper juste avant la hauteur finale du dendogramme
plt.axhline(y=seuil, color='red', linestyle='--', label=f'Seuil = {seuil}') # ajoute une ligne horizontale
plt.legend()
plt.grid(True)
plt.show()

# Créer les groupes à partir du seuil
groupes = fcluster(linked, t=seuil, criterion='distance') # Former des groupes 
# Affichage des groupes formés, ce sont les classes obtenues
print("Groupes obtenus (avec seuil de distance =", seuil, ") :")
for i, nom in enumerate(noms):
    print(f"{nom} → Groupe {groupes[i]}")

# 8.

