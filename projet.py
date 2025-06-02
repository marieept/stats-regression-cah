print("test")
print("test2")
print("test3")
#########################################
#               Partie 2                #
#########################################

import numpy as np
import matplotlib.pyplot as plt

# Données
points = [(1,1), (1,2), (1,5), (3,4), (4,3), (6,2), (0,4)]
x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])


# 2.1 Calcul des Coefficients de Régression

# Moyennes
x_moy = np.mean(x)
y_moy = np.mean(y)

# Coefficients
b1 = np.sum((x - x_moy) * (y - y_moy)) / np.sum((x - x_moy)**2)
b0 = y_moy - b1 * x_moy

# Modèle de prédictions
y_estime = b0 + b1 * x


# 2.3 Coefficient de Détermination R²

# R²
SCE = np.sum((y - y_estime) ** 2)
SCT = np.sum((y - y_moy) ** 2)
SCR = np.sum((y_estime - y_moy) ** 2)
R2 = 1 - SCE / SCT


# 2.2 Visualisation de la Droite de Régression

# Affichage
plt.scatter(x, y, label='Points')
plt.plot(x, y_estime, color='red', label=f'Droite de régression: y = {b0:.2f} + {b1:.2f}x')
plt.legend()
plt.title(f'Droite de régression linéaire simple (R² = {R2:.2f})')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
plt.savefig("figure_2_1.jpg")
print("Graphique enregistré dans le fichier figure_2_1.jpg")



