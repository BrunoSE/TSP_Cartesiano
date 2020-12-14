from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpBinary
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
global logger
global file_format


Coordenadas = pd.read_csv('Coordenadas.csv', index_col=0)
Coordenadas['XY'] = list(zip(Coordenadas['X'], Coordenadas['Y']))
Coordenadas.drop(['X', 'Y'], axis=1, inplace=True)
Nodos = Coordenadas.index.tolist()
Coordenadas = Coordenadas.to_dict('index')
Coord_array = []
for key in Coordenadas:
    Coord_array.append(Coordenadas[key]['XY'])
    Coordenadas[key] = Coordenadas[key]['XY']

MatrizD = cdist(Coord_array, Coord_array, 'euclidean')
MatrizD = pd.DataFrame(data=MatrizD, index=Nodos, columns=Nodos)


def mantener_log():
    global logger
    global file_format
    logger = logging.getLogger(__name__)  # P: número de proceso, L: número de línea
    logger.setLevel(logging.DEBUG)  # deja pasar todos desde debug hasta critical
    print_handler = logging.StreamHandler()
    print_format = logging.Formatter('[{asctime:s}] {levelname:s} L{lineno:d}: {message:s}',
                                     '%Y-%m-%d %H:%M:%S', style='{')
    file_format = logging.Formatter('[{asctime:s}] {processName:s} P{process:d}@{name:s} ' +
                                    '${levelname:s} L{lineno:d}: {message:s}',
                                    '%Y-%m-%d %H:%M:%S', style='{')
    print_handler.setLevel(logging.INFO)
    print_handler.setFormatter(print_format)
    logger.addHandler(print_handler)
    return logger


def traducir_solucion(soluc):
    r = {}
    for var in soluc:
        if var.name[0] == 'x':
            r[(var.name[-2], var.name[-1])] = var.value()
    return r


def dibujar_red(sol):
    sol_matriz = MatrizD.copy()
    for s in sol:
        sol_matriz.loc[s[0], s[1]] = sol[s]

    # G = nx.from_pandas_adjacency(MatrizD, create_using=nx.DiGraph)
    G = nx.from_pandas_adjacency(sol_matriz, create_using=nx.DiGraph)
    nx.set_node_attributes(G, Coordenadas, 'pos')

    # edge_colors = ['red' if sol[e] == 1 else 'white' for e in G.edges]
    # edge_color=edge_colors
    nx.draw(G, Coordenadas, with_labels=True, arrowsize=20, width=1.1)
    plt.show()


def resolver():
    # Create the model
    model = LpProblem(name="Vendedor_viajero", sense=LpMinimize)
    # Initialize the decision variables
    if len(MatrizD.index) != len(MatrizD.columns):
        logger.debug("Error matriz de distancia no es cuadrada!")
        exit()

    nombre_nodos = MatrizD.columns
    n = len(MatrizD.index)

    maxima_distancia = max(MatrizD.max())
    Distancias = MatrizD.to_numpy()
    np.fill_diagonal(Distancias, maxima_distancia)  # llenar diagonal con un costo no-nulo

    x = {}
    for i in range(n):
        for j in range(n):
            lowerBound = 0
            upperBound = 1

            # Prohibir loops
            if i == j:
                upperBound = 0

            # Variables de decision y su restriccion de naturaleza binaria
            x[i, j] = LpVariable(name=f'x_{nombre_nodos[i]}{nombre_nodos[j]}',
                                 lowBound=lowerBound,
                                 upBound=upperBound,
                                 cat=LpBinary)

    u = {}
    for i in range(1, n):
        u[i] = LpVariable(name=f'u_{nombre_nodos[i]}',
                          lowBound=0,
                          upBound=(n - 2),
                          cat='Continuous')
    for i in range(n):
        model += (lpSum([x[i, j] for j in range(n) if i != j]) == 1, f'restriccion de salida {nombre_nodos[i]}')
    for j in range(n):
        model += (lpSum([x[i, j] for i in range(n) if i != j]) == 1, f'restriccion de entrada {nombre_nodos[j]}')

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model += (lpSum([u[i] - u[j] + (n - 1) * x[i, j]]) <= (n - 2), f'restriccion de subtour {nombre_nodos[i]} {nombre_nodos[j]}')

    # Add the objective function to the model
    model += lpSum([Distancias[i][j] * x[i, j] for i in range(n) for j in range(n)])
    logger.debug(model)
    # exit()
    model.solve()

    logger.debug(f"Solver: {model.solver}")

    logger.info(f"Estado solucion: {LpStatus[model.status]}")
    logger.debug(f"Funcion Objetivo: {model.objective.value()}")
    logger.debug(f"Variables:")
    for var in model.variables():
        logger.debug(f"{var.name}: {var.value()}")

    logger.debug(f"Restricciones:")
    for name, constraint in model.constraints.items():
        logger.debug(f"{name}: {constraint.value()}")

    return model.variables()


def main():
    global logger
    logger = mantener_log()
    file_handler = logging.FileHandler(f'TSP.log')

    # en consola loggea desde .debug, en archivo desde .debug hasta .critical
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    solucion = resolver()
    solucion = traducir_solucion(solucion)
    dibujar_red(solucion)


if __name__ == '__main__':
    main()
