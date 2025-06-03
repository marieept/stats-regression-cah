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
plt.savefig("figure_1_2.jpg")

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
    print("La distance euclidienne vaut : ", distance)
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
    mais l’augmentation de la variance intra-classe qu'engendrerait la fusion de deux groupes.
    Elle favorise des regroupements compacts, homogènes, et tend à éviter que des points trop éloignés 
    soient prématurément fusionnés. C’est pourquoi elle est souvent utilisée dans la Classification Ascendante Hiérarchique (CAH),
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

# Tracé du graphique de base 
plt.scatter(x, y, color='blue')
for i in range(len(points)):
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i])
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")

# Récupération des coordonnées des 2 points de la paire la plus proche et calcul de la distance qui les sépare
pair_min, d_min = dist_min(points, dist)
print("La paire de points les plus proche est : ", pair_min)
x_vals = [pair_min[0][0], pair_min[0][1]]
y_vals = [pair_min[1][0], pair_min[1][1]]

# Relier/Encadrer les 2 points les plus proches (Classe Γ₁)
plt.plot(x_vals, y_vals, 'ro--', label="Classe Γ₁")
plt.scatter(x_vals, y_vals, color='red')
plt.title("Regroupement en classes")
plt.grid(True)
plt.xlim(-1, 7)
plt.ylim(0, 6)

# plt.savefig("figure_5_3.jpg")
# print("Graphique enregistré dans le fichier figure_5_3.jpg")


# 5.4.
# Points de Gamma1 (groupe initial)
# dist_min retourne un couple de points (p1, p2)
p1, p2 = pair_min

# retrouver les indices de ces points dans la liste points
i = points.index(p1)
j = points.index(p2)

Gamma1 = [points[i], points[j]]
Gamma1_nom = "G1"

# Points restants
reste_points = []
reste_noms = []
for k in range(n):
    if k != i and k != j:
        reste_points.append(points[k])
        reste_noms.append(noms[k])

# Fonction : distance entre un point et un segment [A,B]
'''calcule la distance au carré entre un point P(px, py) et
un segment défini par les points A(ax, ay) et B(bx, by)'''
def distance_point_segment(px, py, ax, ay, bx, by):
    #Composantes du vecteur AB
    ABx = bx - ax
    ABy = by - ay

    #Composantes du vecteur AP
    APx = px - ax
    APy = py - ay

    #Carré de la longueur du segment AB
    ab2 = ABx**2 + ABy**2
    if ab2 == 0: #si A et B sont confondus, alors c'est une distance avec un point
        return (px - ax)**2 + (py - ay)**2
    
    #Projection orthogonale
    t = (APx * ABx + APy * ABy) / ab2
    if t < 0: #la projection est avant A donc A est le plus proche
        closest_x, closest_y = ax, ay
    elif t > 1: #la projection est après B donc B est le plus proche
        closest_x, closest_y = bx, by
    else:
        closest_x = ax + t * ABx
        closest_y = ay + t * ABy

    #On retourne la distance au carré entre P et le point le plus proche du segment AB
    dx = px - closest_x
    dy = py - closest_y

    return dx**2 +dy**2

# Matrice 6x6 : Gamma1 + 5 autres points
noms_total = [Gamma1_nom] + reste_noms
taille = len(noms_total)
matrice_2 = np.zeros((taille, taille))

for a in range(taille):
    for b in range(taille):
        if a == b: #la matrice est nulle sur la diagonale
            matrice_2[a][b] = 0
        elif a == 0: #distance entre le point et Gamma 1 (distance segment-point)
            px, py = reste_points[b - 1]
            ax, ay = Gamma1[0]
            bx, by = Gamma1[1]
            matrice_2[a][b] = distance_point_segment(px, py, ax, ay, bx, by)
        elif b == 0: #distance entre Gamma 1 et le point (distance segment-point)
            px, py = reste_points[a - 1]
            ax, ay = Gamma1[0]
            bx, by = Gamma1[1]
            matrice_2[a][b] = distance_point_segment(px, py, ax, ay, bx, by)
        else: #distance entre deux points
            matrice_2[a][b] = dist(reste_points[a - 1],reste_points[b - 1])

# Affichage final
print("Matrice avec Gamma1 et les autres points (distances au carré) \n")
df = pd.DataFrame(matrice_2, index=noms_total, columns=noms_total)
print(df.round(1))

#Trouver Gamma2

'''trouver la distance minimale dans une matrice, ainsi que les points correspondants
sauf sur la diagonale'''
def trouver_distance_minimale(matrice, noms):
    min_dist = float('inf')
    indices = (-1, -1)
    
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            if i != j and matrice[i][j] < min_dist:
                min_dist = matrice[i][j]
                indices = (i, j)
    
    nom_i = noms[indices[0]]
    nom_j = noms[indices[1]]
    return nom_i, nom_j, min_dist


p1_min, p2_min, d_min = trouver_distance_minimale(matrice_2, noms_total)

# retrouver les indices de ces points dans la liste points
i = noms_total.index(p1_min)
j = noms_total.index(p2_min)

# p1_min et p2_min sont des noms
Gamma2_noms = [p1_min, p2_min]
Gamma2 = []           # les points de Gamma2
reste_points2 = []    # les autres points
reste_noms2 = []      # leurs noms

# On boucle sur noms et points restants (sauf G1 déjà groupé)
for i in range(1, len(noms_total)):  # on saute "G1", qui est déjà dans Gamma1
    nom = noms_total[i]
    pt = reste_points[i - 1]  # car reste_points n'inclut pas "G1"
    
    if nom in Gamma2_noms:
        Gamma2.append(pt)
    else:
        reste_points2.append(pt)
        reste_noms2.append(nom)

# Construction de la matrice 3 (G1, G2, autres points restants)
groupes = [Gamma1, Gamma2]  # listes de 2 points chacun
noms_total2 = ["G1", "G2"] + reste_noms2
tous_points = groupes + [[p] for p in reste_points2]  # chaque point devient un "groupe d’un point"
taille = len(tous_points)

matrice_3 = np.zeros((taille, taille))

for i in range(taille):
    for j in range(taille):
        if i == j: #sur la diagonale
            matrice_3[i][j] = 0
        else:
            # Si un des deux est un groupe (contient 2 points), on fait la distance segment-point
            if len(tous_points[i]) == 2 and len(tous_points[j]) == 1:
                px, py = tous_points[j][0]

                #i est le segment
                ax, ay = tous_points[i][0]
                bx, by = tous_points[i][1]

                matrice_3[i][j] = distance_point_segment(px, py, ax, ay, bx, by)
            elif len(tous_points[i]) == 1 and len(tous_points[j]) == 2:
                px, py = tous_points[i][0]

                #j est le segment
                ax, ay = tous_points[j][0]
                bx, by = tous_points[j][1]

                matrice_3[i][j] = distance_point_segment(px, py, ax, ay, bx, by)
            else:
                # distance entre deux points
                x1, y1 = tous_points[i][0]
                x2, y2 = tous_points[j][0]
                matrice_3[i][j] = (x1 - x2)**2 + (y1 - y2)**2

            matrice_2[a][b] = (x1 - x2)**2 + (y1 - y2)**2

# Affichage final
print("Matrice avec Gamma2 et les autres points (distances au carré) \n")
df = pd.DataFrame(matrice_3, index=noms_total2, columns=noms_total2)
print(df.round(1))

#Tracé

'''récupère les coordonnées des points sachant qu'on connait leurs noms'''
def coord_from_nom(nom, groupes_dict):
    if nom in groupes_dict:
        return groupes_dict[nom][0]
    else:
        return points[noms.index(nom)]

groupes_dict = {"G1": Gamma1, "G2": Gamma2}

x1, y1 = coord_from_nom(p1_min, groupes_dict)
x2, y2 = coord_from_nom(p2_min, groupes_dict)

x_vals2 = [x1, x2]
y_vals2 = [y1, y2]

plt.plot(x_vals2, y_vals2, 'ro--', label="Classe Γ2")
plt.scatter(x_vals2, y_vals2, color='red')


# 5.5.

# Trouver Gamma3 (deux éléments les plus proches)
p3_min, p4_min, d_min3 = trouver_distance_minimale(matrice_3, noms_total2)

Gamma3_noms = [p3_min, p4_min]
Gamma3 = []

reste_points3 = []
reste_noms3 = []

for i in range(len(noms_total2)):
    nom = noms_total2[i]
    pt = tous_points[i]  # soit un groupe, soit un point (encapsulé dans une liste)

    if nom in Gamma3_noms:
        Gamma3 += pt  # ajoute les points au nouveau groupe
    else:
        reste_points3.append(pt)
        reste_noms3.append(nom)

noms_total3 = ["G3"] + reste_noms3
tous_points3 = [Gamma3] + reste_points3

taille4 = len(tous_points3)
matrice_4 = np.zeros((taille4, taille4))

for i in range(taille4):
    for j in range(taille4):
        if i == j:
            matrice_4[i][j] = 0
        else:
             # Si un des deux est un groupe (contient 2 points), on fait la distance segment-point
            if len(tous_points3[i]) == 2 and len(tous_points3[j]) == 1:
                px, py = tous_points3[j][0]

                #i est le segment
                ax, ay = tous_points3[i][0]
                bx, by = tous_points3[i][1]

                matrice_4[i][j] = distance_point_segment(px, py, ax, ay, bx, by)

            elif len(tous_points3[i]) == 1 and len(tous_points3[j]) == 2:
                px, py = tous_points3[i][0]

                #j est le segment
                ax, ay = tous_points3[j][0]
                bx, by = tous_points3[j][1]
                matrice_4[i][j] = distance_point_segment(px, py, ax, ay, bx, by)
            else:
                #distance entre deux points
                x1, y1 = tous_points3[i][0]
                x2, y2 = tous_points3[j][0]

                matrice_4[i][j] = (x1 - x2)**2 + (y1 - y2)**2


df = pd.DataFrame(matrice_4, index=noms_total3, columns=noms_total3)
print("Matrice avec Gamma3 et les autres points :\n")
print(df.round(1))

groupes_dict = {"G1": Gamma1, "G2": Gamma2, "G3": Gamma3}

x1, y1 = coord_from_nom(p3_min, groupes_dict)
x2, y2 = coord_from_nom(p4_min, groupes_dict)

x_vals3 = [x1, x2]
y_vals3 = [y1, y2]

plt.plot(x_vals3, y_vals3, 'o--', color='orange', label="Classe Γ3")
plt.scatter(x_vals3, y_vals3, color='orange')
plt.show()

# 5.6.

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

# 5.7.
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

# 5.8.

