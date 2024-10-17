## Librerías
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

"""## Carga y lectura de datos"""

url = 'https://raw.githubusercontent.com/HeidyRemache/Database/main/casas.csv'
datos = np.loadtxt(url, delimiter = ",") #Ejecutar desde numpy

"""##Identificar Cluster"""

clusters = DBSCAN(eps = 2, min_samples = 10).fit_predict(datos)
clusters

plt.figure(figsize = (8,8))

plt.scatter(datos[:, 0], datos[:, 1], c = clusters, s = 100)
plt.xlabel("Años de construcción de la casa")
plt.ylabel("Precio")
plt.box(False)
plt.show()
