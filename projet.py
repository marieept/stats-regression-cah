import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
    """Distance euclidienne"""
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    print("La distance euclidienne vaut : ", distance)
    return distance

# 1.b
def dist1(p1, p2):
    """Distance de Manhattan"""
    distance = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    print("La distance de Manhattan vaut : ", distance)
    return distance

# 1.c
def dist_inf(p1, p2):
    """Distance de Chebyshev (max)"""
    distance = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    print("La distance de Chebyshev vaut : ", distance)
    return distance

# 1.d : Distance de Ward ???


# 2.
'''
La fonction dist_min est une fonction permettant de retourner un couple de point situés à une distance minimale l'un de l'autre
paramètre
    tableau : un tableau de points
    dist_func : la fonction du calcul de la distance entre deux points que l'on veut utiliser
'''
def dist_min(tableau, dist_func):
    min_d = float('inf')
    couple_points = None
    for i in range(len(tableau)):
        for j in range(i+1, len(tableau)):
            d = dist_func(tableau[i], tableau[j])
            if d < min_d:
                min_d = d
                couple_points = (tableau[i], tableau[j])
    return couple_points, min_d


# 3.

# Tracé
plt.scatter(x, y, color='blue')
for i in range(len(points)):
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i])
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# Initialisation de la matrice
n = len(x)
matrice_1 = np.zeros((n, n))

# Remplissage de la matrice avec d²
for i in range(n):
    for j in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        d_squared = (xi - xj)**2 + (yi - yj)**2
        matrice_1[i][j] = d_squared

# Affichage avec pandas pour lisibilité
data = pd.DataFrame(matrice_1, index=x, columns=x)
print("Matrice des distances euclidiennes au carré :\n")
print(data.round(1))

'''
# Encadrer M1 et M7 (Classe Γ₁)
x_vals = [points[0][0], points[6][0]]
y_vals = [points[0][1], points[6][1]]
#plt.plot(x_vals, y_vals, 'ro--')
plt.scatter(x_vals, y_vals, color='red')
plt.title("Regroupement initial : Classe Γ₁ = {M1, M7}")
plt.grid(True)
plt.xlim(-1, 7)
plt.ylim(0, 6)
plt.show()

plt.savefig("figure_5_3.jpg")
print("Graphique enregistré dans le fichier figure_5_3.jpg")
'''

#4.

# Trouver le couple le plus proche
min_dist = float('inf') #valeur très grande (infini)
min_pair = (0, 0) 

min_pair, min_dist = dist_min(points, dist)

# Points de Gamma1 (groupe initial)
# dist_min retourne un couple de points (p1, p2)
p1, p2 = min_pair

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
            x1, y1 = reste_points[a - 1]
            x2, y2 = reste_points[b - 1]
            matrice_2[a][b] = (x1 - x2)**2 + (y1 - y2)**2

# Affichage final
print("Matrice avec Gamma1 et les 5 autres points (distances au carré) \n")
df = pd.DataFrame(matrice_2, index=noms_total, columns=noms_total)
print(df.round(1))
