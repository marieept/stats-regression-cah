import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import math
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

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
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i]) #décaler le texte du point
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
plt.savefig("figure_1_2.jpg")

#   Interprétation statistique :

# Les valeurs x sont plus dispersées que les valeurs y.
# La moyenne de x est plus faible que celle de y, indiquant une asymétrie spatiale.
# L’étendue plus grande pour x traduit une plus grande variabilité sur l’axe horizontal.

#   Interprétation graphique :

# Le nuage montre une certaine tendance croissante mais non strictement linéaire.
# Il est possible qu’une relation linéaire approximative existe entre x et y, à tester dans la partie suivante.


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
plt.title(f'Droite de régression linéaire simple (R² = {R2:.4f})')
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

# 3.1. Résidus et somme des carrés des erreurs (SCE)
e = y - y_pred

SCE=0

for i in range(len(points)):
    SCE += e[i]**2 
print("SCE (Somme des carrés des erreurs) :", SCE)

# 3.2. Etimation de la variance des erreurs : MSE
n= len(x)

MSE = SCE / (n-2)

print("MSE (Erreur quadratique moyenne) :", MSE)

#Le n-2 vient du fait qu'on a estimé 2 paramètres (b0 et b1). On parle alors de degrés de liberté.

# 3.3. Ecart-type des erreurs
s = np.sqrt(MSE)

print("Écart-type des erreurs :", s)

# 3.4. Interprétation des Résultats

# Affichage d'un tableau pour mieux visualiser nos résultats
# Création du tableau
Tab_Affichage = pd.DataFrame({
    "x": x,
    "y (observée)": y,
    "ŷ (prédite)": np.round(y_pred, 4),
    "Résidu (y - ŷ)": np.round(e, 4)
})

print(Tab_Affichage)

#     - Interprétation des coefficients

#     b0=3.33 (ordonnée à l'origine) : c'est la valeur estimée de y quand x=0. Cela signifie qu'à l'origine de l'axe x, la droite de régression prévoit y=3.33.

#     b1=−0.15 (pente) : chaque augmentation de 1 unité en x entraîne une baisse moyenne de y de 0.15. La relation est donc légèrement décroissante, mais très faible.

#     - Coefficient de détermination R2

#     Le R2=0.049 (soit 4.9%) indique que seulement 4.9% de la variation de y est expliquée par la variable x.

#     Cela signifie que la droite de régression explique très peu la variabilité des points. La majorité de la variation de y provient donc d'autres facteurs non capturés par ce modèle.

#     - Analyse des erreurs

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

# Statistique de test pour b1

t_b1 = (b1-0)/SEb1 # t suit une loi de Student donc on calcul la p-valeur pour savoir si on rejette l'hypothèse ou non
p_b1 = 2 * (1 - stats.t.cdf(abs(t_b1), df=n - 2)) #fonction de répartition cumulative
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

# Statistique de test pour b0

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

t_crit = stats.t.ppf(1-alpha/2, df= n-2) #CDF inverse ->  sert à trouver la valeur critique pour un test bilatéral basé sur la loi de Student.

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

# 5.1.a.
def dist(p1, p2):
    '''
    Cette fonction prend deux couples de points en paramètre et retourne une distance entre ces deux points
    :param p1 : premier couple de point
    :param p2 : second couple de point
    :return : la distance entre ces deux points
    '''
    """Distance euclidienne"""
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    #print("La distance euclidienne vaut : ", distance)
    return distance

# 5.1.b.
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

# 5.1.c.
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

# 5.1.d. : Distance de Ward ???

'''
        La distance de Ward ne mesure pas une simple distance géométrique entre deux points, 
    mais l'augmentation de la variance intra-classe qu'engendrerait la fusion de deux groupes.
    Elle favorise des regroupements compacts, homogènes, et tend à éviter que des points trop éloignés 
    soient prématurément fusionnés. C'est pourquoi elle est souvent utilisée dans la Classification Ascendante Hiérarchique (CAH),
    notamment pour éviter la formation de groupes déséquilibrés.
    
    Distance de Ward entre un groupe 1 et un groupe 2 :
    
        ((len(gp1)*len(gp2))/(len(gp1)+len(gp2)))*(np.abs(np.mean(gp1)-np.mean(gp2))**2)
'''


# 5.2.
def dist_min(tableau, dist_func):
    '''
    Cette fonction prend un tableau et une fonction en paramètre et retourne un couple de point situés à une distance minimale l'un de l'autre
    :param : tableau : un tableau de points
    :param : dist_func : la fonction du calcul de la distance entre deux points que l'on veut utiliser
    :return : un couple de points et la distance qui les sépare
    '''
    min_d = float('inf')
    couple_points = None
    for i in range(len(tableau)):
        for j in range(i+1, len(tableau)):                  #Permet de parcourir toutes les lignes et toutes les colonnes d'un tableau 
            d = dist_func(tableau[i], tableau[j])           #Récupère la distance à partir de la fonction donnée en entrée
            if d < min_d:                                   #Calcul de toutes les distances entre les points et garde la distance minimum
                min_d = d
                couple_points = (tableau[i], tableau[j])    #Couple de points le plus proche
    return couple_points, min_d                             


# 5.3.

# Initialisation de la matrice
n = len(x)                                                  #Définit la taille de la matrice carré
matrice_1 = np.zeros((n, n))                                #Rempli la matrice carré de taille n x n par des 0

# Remplissage de la matrice avec d²
for i in range(n):                                          #Permet le remplissage d'une ligne
    for j in range(n):                                      #Permet de passer d'une ligne à la suivante
        xi, yi = points[i]                                  #Coordonnées d'une premier point i
        xj, yj = points[j]                                  #Coordonnées d'un deuxième point j
        d_eucli = dist(points[i], points[j])                #Calcul de la distance euclidienne entre ces 2 points
        d_eucli_2 = d_eucli**2                              #Calcul du carré de la distance euclidienne
        matrice_1[i][j] = d_eucli_2                         #Rempli la matrice avec les valeurs obtenues

# Affichage avec pandas pour lisibilité
data = pd.DataFrame(matrice_1, index=y, columns=x)          #Affiche la matrice dans le terminal (avec les coordonnées x et y pour les colonnes et les lignes)
print("Matrice des distances euclidiennes au carré :\n")
print(data.round(1))
print("\n")

# Tracé du graphique de base 
plt.scatter(x, y, color='blue')
for i in range(len(points)):
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i])
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")

# Récupération des coordonnées des 2 points de la paire la plus proche et calcul de la distance qui les sépare
pair_min, d_min = dist_min(points, dist) #[(xi,yi),(xj,yj)]
print("La paire de points les plus proche est : ", pair_min)
x_vals = [pair_min[0][0], pair_min[0][1]] #[xi,xj]
y_vals = [pair_min[1][0], pair_min[1][1]] #[yi,yj]

# Relier/Encadrer les 2 points les plus proches (Classe Γ₁)
plt.plot(x_vals, y_vals, 'go--', label="Classe Γ₁")
plt.scatter(x_vals, y_vals, color='green')
plt.title("Regroupement en classes")
plt.grid(True)
plt.xlim(-1, 7)
plt.ylim(0, 6)
plt.show()

# plt.savefig("figure_5_3.jpg")
# print("Graphique enregistré dans le fichier figure_5_3.jpg")


#5.4

#____________________________________________________________________________
#Fonctions nécessaires aux matrices

def dist_groupe_point(groupe, point):
    '''
    Recherche la distance minimale entre un point et un ensemble de points
    '''
    return min(dist(g, point) for g in groupe)

def dist_groupe_groupe(groupe1, groupe2):
    '''
    Recherche la distance minimale entre deux ensembles de points
    '''
    return min(dist(p1, p2) for p1 in groupe1 for p2 in groupe2)

def construire_matrice(groupes, points_restants):
    '''
    Construit une matrice de distances entre les groupes (listes de points donc liste de liste si plusieurs points)
    et les points restants (points seuls).
    '''
    n_groupes = len(groupes)
    n_restants = len(points_restants)
    taille = n_groupes + n_restants

    matrice = np.zeros((taille, taille))
    
    # Distances entre groupes
    for i in range(n_groupes):
        for j in range(i + 1, n_groupes): #on boucle que sur la partie supérieure car la matrice est symétrique
            d = dist_groupe_groupe(groupes[i], groupes[j])
            matrice[i, j] = d
            matrice[j, i] = d #on remplit la partie inférieure

    # Distances entre groupe et point
    for i, groupe in enumerate(groupes): #ajoute un index automatique et parcourt les points du groupe
        for j, point in enumerate(points_restants):
            d = dist_groupe_point(groupe, point) #groupe prend un point du groupe et point parmi ceux restants
            matrice[i, n_groupes + j] = d #i correspond à l'indice de la ligne du groupe
            matrice[n_groupes + j, i] = d #i correspond à l'indice de la colonne du groupe

    # Distances entre points restants
    for i in range(n_restants):
        for j in range(i + 1, n_restants):
            d = dist(points_restants[i], points_restants[j])
            matrice[n_groupes + i, n_groupes + j] = d
            matrice[n_groupes + j, n_groupes + i] = d

    return matrice

def find_min(matrice):
    '''
    Trouve la paire (i, j) avec la plus petite valeur dans la matrice,
    en considérant uniquement les éléments au-dessus de la diagonale (j > i) 
    comme la matrice est symétrique
    '''
    n = matrice.shape[0] #nombre de lignes dans la matrice = nombre colonnes car symétrique

    min_dist = float('inf') #on initialise le minimum à l'infini pour commencer
    fusion_indices = (None, None) #on intialiser les indices
    
    #Parcours de la matrice
    for i in range(n):
        for j in range(i + 1, n):
            if matrice[i, j] < min_dist:
                min_dist = matrice[i, j]
                fusion_indices = (i, j)
                
    return fusion_indices


def get_groupe_ou_point(idx, points_isoles):
    '''
    Récupère si c'est un groupe ou un point isolé à partir d'un indice dans la matrice
    car les groupes sont les premiers indices dans la matrice
    '''
    if idx < len(groupes):
        return groupes[idx]
    else:
        return [points_isoles[idx - len(groupes)]]
#____________________________________________________________________________
#matrice avec GAMMA 1

classe_G1= list(pair_min)
print("\n Points dans Γ1 :", classe_G1, "\n")

#Points non encore dans un groupe
points_restants=[p for p in points if p not in classe_G1]

# Utilisation de la fonction générique
matrice_2 = construire_matrice([classe_G1], points_restants)

# Création des noms pour les lignes/colonnes
noms_groupes = ["Γ1"] + [f"{noms[points.index(p)]}" for p in points_restants] #trouve l’indice du point p dans la liste points pour trouver le nom du point

#Affichage avec pandas
df2 = pd.DataFrame(matrice_2, index=noms_groupes, columns=noms_groupes)
print("Matrice des distances euclidiennes au carré avec Γ1 :\n")
print(df2.round(1))
print("\n")


#GAMMA 2

# Trouver la paire avec la distance minimale
fusion_indices = find_min(matrice_2)

# On construit la liste des points qui forment Γ2
if 0 in fusion_indices: #si un des deux indices vaut 0, ça correspond à Γ1 dans la matrice

    #Récupération de l'indice du point isolé

    if fusion_indices[0]==0:#si Γ1 est le premier des deux indices
        autre_idx = fusion_indices[1]
    else:
        autre_idx = fusion_indices[0]

    classe_G2 = classe_G1 + [points_restants[autre_idx - 1]] #-1 car la matrice a un indice d'avance car elle comporte Γ1

else: #si les deux indices sont des points isolés

    classe_G2 = [points_restants[fusion_indices[0] - 1], points_restants[fusion_indices[1] - 1]]

print("Points dans Γ2 :", classe_G2, "\n")

#Tracé sur le graphique de la classe Γ2

# On récupère les indices des deux groupes à fusionner
i1, i2 = fusion_indices

# Si l'un est Γ1 (indice 0), on récupère le point isolé fusionné
if i1 == 0:
    p1 = classe_G1[0]
    p2 = points_restants[i2 - 1]

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

plt.plot(x_vals2, y_vals2, 'o--', color='orange', label="Classe Γ2")
plt.scatter(x_vals2, y_vals2, color='orange')

# 5.5.

# matrice avec GAMMA 2

points_restants2 = [p for p in points if p not in classe_G1 and p not in classe_G2]

# Regrouper les groupes existants dans une liste
groupes = [classe_G1, classe_G2]

# Utiliser la fonction existante pour construire la matrice
matrice_3 = construire_matrice(groupes, points_restants2)

noms_groupes2 = ["Γ1", "Γ2"] + [f"{noms[points.index(p)]}" for p in points_restants2]

df3 = pd.DataFrame(matrice_3, index=noms_groupes2, columns=noms_groupes2)
print("Matrice des distances euclidiennes au carré avec Γ1 et Γ2 :\n")
print(df3.round(1))
print("\n")

#GAMMA 3

# On suppose que classe_G1 et classe_G2 sont déjà définies (Γ1 et Γ2)
groupes = [classe_G1, classe_G2]
points_isoles = points_restants2  # points encore non groupés après Γ2

# Trouver la paire avec la distance minimale (hors diagonale 0)
fusion_indices = find_min(matrice_3)

# Fusion des deux entités (groupes ou points)
classe_G3 = get_groupe_ou_point(fusion_indices[0], points_restants2) + get_groupe_ou_point(fusion_indices[1], points_restants2)

print("Points dans Γ3 :", classe_G3, "\n")

# Tracer la ligne entre les deux éléments fusionnés
p1 = get_groupe_ou_point(fusion_indices[0], points_restants2)[0]
p2 = get_groupe_ou_point(fusion_indices[1], points_restants2)[0]

x_vals3 = [p1[0], p2[0]]
y_vals3 = [p1[1], p2[1]]

plt.plot(x_vals3, y_vals3, 'o--', color='red', label="Classe Γ3")
plt.scatter(x_vals3, y_vals3, color='red')

plt.legend()
plt.show()

# matrice avec GAMMA 3

# Liste des points encore non affectés aux groupes Γ1, Γ2, Γ3
points_restants3 = [p for p in points if p not in classe_G1 and p not in classe_G2 and p not in classe_G3]
# Regrouper les groupes existants dans une liste
groupes.append(classe_G3)

# Utiliser la fonction existante pour construire la matrice
matrice_4 = construire_matrice(groupes, points_restants3)

# Noms des groupes + points restants
noms_groupes3 = ["Γ1", "Γ2", "Γ3"] + [f"{noms[points.index(p)]}" for p in points_restants3]

df3 = pd.DataFrame(matrice_4, index=noms_groupes3, columns=noms_groupes3)
print("Matrice des distances euclidiennes au carré avec Γ1, Γ2 et Γ3 :\n")
print(df3.round(1))
print("\n")

#GAMMA 4

# Trouver la paire avec la distance minimale
fusion_indices = find_min(matrice_4)

# Fusion selon les indices
classe_G4 = get_groupe_ou_point(fusion_indices[0], points_restants3) + get_groupe_ou_point(fusion_indices[1], points_restants3)

print("Points dans Γ4 :", classe_G4, "\n")

# matrice avec GAMMA 4

# Retirer les groupes fusionnés de la liste groupes car c'est l'union de deux classes
i1,i2 = fusion_indices

groupes.pop(max(i1, i2))
groupes.pop(min(i1, i2))

# Regrouper les groupes existants dans une liste
groupes.append(classe_G4)

# Liste des points encore non affectés aux groupes Γ1, Γ2, Γ3, Γ4
points_restants4 = [p for p in points if all(p not in g for g in groupes)]

# Utiliser la fonction existante pour construire la matrice
matrice_5 = construire_matrice(groupes, points_restants4)

# Noms des groupes + points restants
noms_groupes4 = ["Γ3", "Γ4"] + [f"{noms[points.index(p)]}" for p in points_restants4]

df4 = pd.DataFrame(matrice_5, index=noms_groupes4, columns=noms_groupes4)
print("Matrice des distances euclidiennes au carré avec Γ3 et Γ4 :\n")
print(df4.round(1))
print("\n")

#GAMMA 5

# Trouver la paire avec la distance minimale 
fusion_indices = fusion_indices = find_min(matrice_5)

# Fusion selon les indices
classe_G5 = get_groupe_ou_point(fusion_indices[0], points_restants4) + get_groupe_ou_point(fusion_indices[1], points_restants4)

print("Points dans Γ5 :", classe_G5, "\n")

# matrice avec GAMMA 5

# Retirer les groupes fusionnés de la liste groupes car c'est l'union de deux classes
i1,i2 = fusion_indices

groupes.pop(max(i1, i2))
groupes.pop(min(i1, i2))

# Regrouper les groupes existants dans une liste
groupes.append(classe_G5)

# Liste des points encore non affectés aux groupes Γ1, Γ2, Γ3, Γ4
points_restants5 = [p for p in points if all(p not in g for g in groupes)]

# Utiliser la fonction existante pour construire la matrice
matrice_6 = construire_matrice(groupes, points_restants5)

# Noms des groupes + points restants
noms_groupes5 = ["Γ5"] + [f"{noms[points.index(p)]}" for p in points_restants5]

df5 = pd.DataFrame(matrice_6, index=noms_groupes5, columns=noms_groupes5)
print("Matrice des distances euclidiennes au carré avec Γ5 :\n")
print(df5.round(1))
print("\n")

#GAMMA 6

# Trouver la paire avec la distance minimale (hors diagonale 0)
fusion_indices = fusion_indices = find_min(matrice_6)

# Fusion selon les indices
classe_G6 = get_groupe_ou_point(fusion_indices[0], points_restants5) + get_groupe_ou_point(fusion_indices[1], points_restants5)

print("Points dans Γ6 :", classe_G6, "\n")

# matrice avec GAMMA 6

# Retirer les groupes fusionnés de la liste groupes car c'est l'union de deux classes
i1, i2 = sorted(fusion_indices, reverse=True) #trier du plus grand au plus petit
for i in (i1, i2):
    if i < len(groupes):  # Ne pas pop s'il s'agit d'un point isolé
        groupes.pop(i)

# Regrouper les groupes existants dans une liste
groupes.append(classe_G6)

# Liste des points encore non affectés aux groupes Γ1, Γ2, Γ3, Γ4
points_restants6 = [p for p in points if all(p not in g for g in groupes)]

# Utiliser la fonction existante pour construire la matrice
matrice_7 = construire_matrice(groupes, points_restants6)

# Noms des groupes + points restants
noms_groupes6 = ["Γ6"] + [f"{noms[points.index(p)]}" for p in points_restants6]

df6 = pd.DataFrame(matrice_7, index=noms_groupes6, columns=noms_groupes6)
print("Matrice des distances euclidiennes au carré avec Γ6 :\n")
print(df6.round(1))
print("\n")

#Visualisation progressive des classes
# Liste de toutes les classes formées au fur et à mesure
classes_etapes = [classe_G1, classe_G2, classe_G3, classe_G4, classe_G5, classe_G6]

# Couleurs pour les classes
couleurs = ['blue', 'orange', 'red', 'green', 'purple', 'brown']

#On extrait les coordonnées x et y des points pour les afficher
x_all = [p[0] for p in points]
y_all = [p[1] for p in points]

#Boucle sur chaque étape (de Γ1 à Γ6)
for i in range(len(classes_etapes)):
    plt.figure(figsize=(8,6)) #on crée une nouvelle figure pour chaque étape
    
    # Tracer tous les points en gris clair
    plt.scatter(x_all, y_all, color='lightgray', label='Points initiaux')
    
    # Tracer toutes les classes formées jusqu'à l'étape i incluse
    for j in range(i+1):
        classe = classes_etapes[j] #on récupère la classe Γ(j+1)
        x_c = [p[0] for p in classe] #coordonnées x de la classe
        y_c = [p[1] for p in classe] #coordonnées y de la classe
        
        #affichage des points de la classe et de la ligne en pointillés
        plt.scatter(x_c, y_c, color=couleurs[j], s=100, label=f"Classe Γ{j+1}")
        plt.plot(x_c, y_c, linestyle='--', color=couleurs[j])
    
    #on ajoute les titres et les axes
    plt.title(f"Fusion jusqu'à la classe Γ{i+1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()



# 5.6.

# Étape 1 : calcul de la matrice de linkage avec méthode "single"
# La fonction linkage construit la hiérarchie de regroupement en utilisant la distance euclidienne.
# Ici, on utilise la méthode 'single' (plus proche voisin), ce qui signifie que la distance entre deux groupes est définie comme la distance minimale entre un point de l'un et un point de l'autre.
# Cette ligne automatise toutes les étapes de fusion des groupes jusqu'à ce qu'il n'en reste plus qu'un.


linked = linkage(points, method='single', metric='euclidean') # La variable linked contient toutes les étapes de regroupement.

linked_df = pd.DataFrame(linked, columns=["Groupe 1", "Groupe 2", "Distance", "Nb points"])
print(linked_df)

# Étape 2 : Affichage du dendrogramme
# La fonction dendrogram() affiche visuellement les fusions successives entre groupes.
# Chaque branche représente une étape de regroupement ; plus elle est haute, plus la distance entre les groupes fusionnés est grande.
# Les labels permettent de visualiser quels points (M1 à M7) sont fusionnés à chaque étape.
# Cela remplace et résume graphiquement toutes les étapes manuelles de la classification (Γ1, Γ2, etc.).

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=noms) #trace l’arbre des regroupements successifs, label=noms permet d’afficher les noms des points
plt.title('Dendrogramme - CAH (single linkage)')
plt.xlabel('Points')
plt.ylabel('Distance euclidienne') #l'axe verical représente la distance entre groupes au moment de leur fusion
plt.grid(True)
plt.show()

# 5.7.
# On repart du dendrogramme que l'on a construit dans la question 6 
plt.figure(figsize=(10, 6))
dendrogram(linked, labels=noms)
plt.title('Dendrogramme - CAH (single linkage)')
plt.xlabel('Points')
plt.ylabel('Distance euclidienne')
seuil = 2 # seuil de découpe, on ne peut pas mettre au-dessus de la hauteur finale du dendrogramme
# le seuil de découpe permet de former des groupes, il faut couper juste avant la hauteur finale du dendrogramme
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

# 5.8.
# on charge les données du fichier
df = pd.read_excel("Data_PE_2025-CSI3_CIR3.xlsx")
print(df.head())

# On supprime la première ligne vide si elle existe
df = df.dropna(how='all') #supprime uniquement les lignes où TOUTES les valeurs sont manquantes

# On sépare les colonnes des noms et des variables numériques
noms_tab = df.iloc[:, 0] #première colonne, toutes les lignes
data = df.iloc[:, 1:].astype(float)  #on convertit en float pour éviter les erreurs

# On convertit les colonnes en float (valeurs numériques)
#coerce : si une valeur ne peut pas être convertie, elle sera remplacée par NaN (données manquantes)
#apply : applique une fonction à chaque colonne
data = data.apply(pd.to_numeric, errors='coerce')

valid_rows = data.dropna() # On supprime les lignes contenant des NaN 
noms_tab2 = noms_tab[valid_rows.index]  # On met a jour les noms des individus pour garder ceux dont les lignes sont valides
data = valid_rows # On remplace 'data' par la version nettoyée sans valeurs manquantes

#  On standardiser les données (centrer-réduire) : chaque colonne aura moyenne 0 et écart-type 1
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
#fit : calculer les paramètres nécessaires à la transformation
#transform : applique la transformation calculée

huit = linkage(data_scaled, method='ward') # On applique la Classification Ascendante Hiérarchique (CAH) avec la méthode de Ward

# On trace le dendrogramme (arbre de regroupement hiérarchique)
plt.figure(figsize=(12, 6))
dendrogram(huit, labels=noms_tab2.tolist(), leaf_rotation=90)  # On affiche les noms des individus
plt.title("Dendrogramme CAH")
plt.xlabel("Individus")
plt.ylabel("Distance")
plt.axhline(y=8, color='r', linestyle='--') # Ligne horizontale suggérant une coupure (choix de nombre de groupes)
plt.tight_layout()
plt.show()

####### A VOIR ##########

# On cherche automatiquement le meilleur nombre de groupes en testant de 2 à 10 groupes
print("\nÉvaluation du coefficient de silhouette pour différents groupes :")
# coefficient de silhouette : mesure de la qualité d’un regroupement 
best_score = -1 # On initialise le meilleur score à une valeur très basse
best_k = None # Variable qui contiendra le meilleur nombre de groupes
for k in range(2, 11): # On teste les valeurs de k de 2 à 10
    labels = fcluster(huit, t=k, criterion='maxclust') # On coupe l'arbre pour obtenir k groupes
    score = silhouette_score(data_scaled, labels) # On calcule le score de silhouette pour évaluer la qualité du regroupement
    print(f"{k} groupes : coefficient de silhouette = {score:.4f}") # Affichage du score
    if score > best_score: # Si ce score est le meilleur qu'on a vu jusque-là
        best_score = score # On le sauvegarde
        best_k = k # On note la valeur de k correspondante

print(f"\nMeilleur nombre de groupes : {best_k} avec un coefficient de silhouette de {best_score:.4f}") # On affiche le meilleur nombre de groupes et le meilleur score obtenu
# Le coefficient de silhouette est de : 0.4907

#########################################
#               Partie 6                #
#########################################

print("\nPARTIE 6\n")

# 6.1. - Indices d'évaluation

# Le Silhouette Score mesure dans quelle mesure un objet est bien attribué à son groupe, par rapport aux autres groupes.
# Pour un point i, on calcule :
#    a(i) : la distance moyenne entre i et tous les autres points du même groupe.
#    b(i) : la plus petite distance moyenne entre i et tous les points des autres groupes (c'est-à-dire le groupe voisin le plus proche).

# Ensuite, on calcule le score silhouette de i :
#   s(i)=b(i)−a(i)max⁡(a(i),b(i))
#   s(i)=max(a(i),b(i))b(i)−a(i)​
# Interprétation :
#    s(i)≈1s(i)≈1 : Le point est bien regroupé, très loin des autres groupes.
#    s(i)≈0s(i)≈0 : Le point est à la frontière entre deux groupes.
#    s(i)<0s(i)<0 : Le point est mal classé, plus proche d’un autre groupe que du sien.
# Silhouette Score global :
#   On calcule la moyenne des s(i)s(i) pour tous les points afin d’évaluer la qualité globale du regroupemennt.
#   Silhouette Score global ≈ 1 : Très bons groupes.
#     ≈ 0 : groupes qui se chevauchent.
#     < 0 : Mauvais regroupement

# 6.2 - Validation croisée

# On compare avec une autre méthode de clustering : k-means
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=0)
    km_labels = km.fit_predict(data_scaled)
    #fit : ajuste le modèle aux données (calcule les centres de clusters)
    #predict : attribue à chaque point une étiquette de cluster
    score = silhouette_score(data_scaled, km_labels)
    print(f"k-means avec {k} groupes : silhouette = {score:.4f}")
print("\n")
# Le coefficient de silhouette avec la méthode k-means est de : 0,2065
# Si on compare les deux scores de silhouettes, 0.4907 > 0,2065 donc la Classification Ascendante Hiérarchique (CAH) 
# produit des groupes plus cohérents et mieux séparés sur ce jeu de données que k-means.

# 6.3 - Interprétation des résultats


# On recrée un DataFrame avec les données propres et les noms des individus
final_labels = fcluster(huit, t=best_k, criterion='maxclust') # on calcul les labels finaux avec le meilleur k 
#maxclust : critère indiquant que t correspond au nombre maximal de clusters souhaité
df_clusters = data.copy()
df_clusters['Nom'] = noms_tab2.values
df_clusters['Cluster'] = final_labels
# On exclut la colonne 'Nom' avant de faire la moyenne des variables par groupe

print("\nCalcul de la moyenne des données")
print(df_clusters.drop(columns=['Nom']).groupby('Cluster').mean())
#supprime la colonne nom -> crée des groupes correspondant à chaque cluster identifié -> calcule la moyenne des colonnes numériques pour chaque groupe

print("\nCalcul de la médiane des données")
print(df_clusters.drop(columns=['Nom']).groupby('Cluster').median())

print("\nCalcul de la variance des données")
print(df_clusters.drop(columns=['Nom']).groupby('Cluster').var())


# 6.5 - Visualisation avancée

#ACP
# on réduit la dimension des données à 2 composantes principales, on conserve le maximum de variance possible
pca = PCA(n_components=2) 
# on applique la transformation ACP sur les données data_scaled (normalisées), et on stocke le résultat dans X_pca
X_pca = pca.fit_transform(data_scaled) 
# X_pca est une matrice 2D (n_lignes × 2 colonnes), représentant chaque individu selon les deux axes principaux d'inertie.
plt.scatter(X_pca[:,0], X_pca[:,1], c=final_labels, cmap='rainbow') # on trace un nuage de point, rainbow : palette de couleurs utilisée
plt.title("Projection ACP + Clusters")
plt.show()

print(pca.explained_variance_ratio_) # on affiche la variance expliquée par les 2 premières composantes


# Chaque point représente un individu.
# Les couleurs indiquent le groupe (cluster) auquel l’individu appartient (issu de la CAH).
# Si les groupes forment des nuages bien séparés, cela indique une bonne qualité de clustering.
# Si certains groupes sont chevauchants ou mélangés, cela peut indiquer une moins bonne séparation ou des groupes proches.
# Ce type de visualisation est très utile pour confirmer visuellement ce que le Silhouette Score a mesuré numériquement : la qualité des regroupements.

#t-SNE 
tsne = TSNE(n_components=2, perplexity=10, random_state=42) # on réduit à 2 dimensions, perplexity contrôle le voisinage, random_state fixe l’aléatoire pour reproduire les mêmes résultats à chaque exécution.
X_tsne = tsne.fit_transform(data_scaled) # on applique la transformation t-SNE sur les données normalisées.
# X_tsne est un tableau avec 2 colonnes, représentant les deux nouvelles dimensions
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=final_labels, cmap='rainbow') # on trace un nuage de points
plt.title("Projection t-SNE + Clusters")
plt.show()

#heatmatap
# Calcul de la matrice des distances euclidiennes
distance_matrix = squareform(pdist(data_scaled, metric='euclidean')) #pdist(...) : calcule les distances entre chaque paire d’individus dans data_scaled
# squareform(...) : convertit le vecteur des distances en matrice symétrique (n × n), où chaque case [i][j] correspond à la distance entre l’individu i et l’individu j.

indices_tries= np.argsort(final_labels) # on récupère les indices triés par cluster question 6.3
matrice_triee = distance_matrix[indices_tries, :][:, indices_tries] # on trie la matrice des distances selon ces indices

# Affichage de la heatmap
plt.figure(figsize=(10, 8))
# carte de chaleur
# distance_matrix : matrice des distances à visualiser.
sns.heatmap(matrice_triee, cmap='viridis') # cmap='viridis' : palette de couleurs (du violet au jaune) → plus la distance est petite, plus la couleur est foncée.
plt.title("Heatmap des distances euclidiennes entre individus")
plt.xlabel("Individus")
plt.ylabel("Individus")
plt.show()

# Les zones sombres (proches de 0) indiquent que les individus sont très similaires (courte distance),
# tandis que les zones claires indiquent des individus très différents.