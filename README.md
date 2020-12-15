# Vendedor Viajero Cartesiano

Script que resuelve el TSP

## Detalles

Este programa lee un .csv con coordenadas en un plano euclideano y luego ocupa la librería PuLP de python para resolver el problema del vendedor viajero con esas coordenadas. Finalmente el script dibuja la solucion en un grafo NetworkX.

```
Coordenadas.csv: corresponde a un csv con las coordenadas X e Y en columnas separadas
```

### Requerimientos
python 3\
pip install pulp\
pip install numpy\
pip install matplotlib\
pip install networkx\
pip install scipy\
pip install pandas\
 \

El solver ocupado por defecto CBC (también conocido como Coin-OR) que viene en la librería PuLP es open-source y gratuito. Se puede usar un solver distinto al que viene por defecto, por ejemplo, una alternativa gratuita es GLPK. Alternativas comerciales más conocidas son Gurobi y CPLEX.
