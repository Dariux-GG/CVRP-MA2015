import math
from pulp import *




#Datos iniciales
nodos = range(11)
clientes = range(1, 11)
m = len(clientes)
vehiculos = range(1, m + 1)


# Matriz de distancias
c = [
    #  0      1      2      3      4      5      6      7      8      9      10
    [0.00, 36.06, 36.06, 26.93, 36.06, 25.50, 22.36, 29.15, 36.40, 11.18, 20.62],
    [36.06, 0.00, 14.14, 40.31, 60.00, 25.50, 44.72, 63.64, 71.59, 26.93, 52.20],
    [36.06, 14.14, 0.00, 30.41, 50.99, 35.36, 50.99, 65.19, 68.01, 25.00, 47.17],
    [26.93, 40.31, 30.41, 0.00, 20.62, 46.10, 49.24, 50.25, 43.01, 21.21, 22.36],
    [36.06, 60.00, 50.99, 20.62, 0.00, 60.42, 56.57, 47.43, 30.41, 36.40, 18.03],
    [25.50, 25.50, 35.36, 46.10, 60.42, 0.00, 21.21, 44.72, 60.21, 25.00, 46.10],
    [22.36, 44.72, 50.99, 49.24, 56.57, 21.21, 0.00, 25.50, 46.10, 30.41, 39.05],
    [29.15, 63.64, 65.19, 50.25, 47.43, 44.72, 25.50, 0.00, 25.00, 40.31, 30.41],
    [36.40, 71.59, 68.01, 43.01, 30.41, 60.21, 46.10, 25.00, 0.00, 44.72, 21.21],
    [11.18, 26.93, 25.00, 21.21, 36.40, 25.00, 30.41, 40.31, 44.72, 0.00, 25.50],
    [20.62, 52.20, 47.17, 22.36, 18.03, 46.10, 39.05, 30.41, 21.21, 25.50, 0.00]
]


# Demandas
d = {
    0: 0,
    1: 12,
    2: 18,
    3: 10,
    4: 15,
    5: 14,
    6: 20,
    7: 16,
    8: 11,
    9: 13,
    10: 17
}

N = list(range(len(c)))
DEPOSITO = 0


# Capacidad
Q = 50

# parametros
time_limit = 60
gap = 0.0001


N_clientes    = [i for i in N if i != DEPOSITO]
total_demanda = sum(d[i] for i in N_clientes)
print(f"[INFO] Clientes a rutear: {len(N_clientes)}, demanda total={total_demanda:.1f}")



#-------------------------------------------------------------------------------------------------
#Modelo
model = LpProblem("CVRP", LpMinimize)
#-------------------------------------------------------------------------------------------------


arcos = [(i, j) for i in N for j in N if i != j]

# variables
x = LpVariable.dicts("x", arcos, 0, 1, LpBinary)
f = LpVariable.dicts("f", arcos, 0)


# OBJETIVO: minimizar distancia total
model += lpSum(c[i][j] * x[(i, j)] for i, j in arcos)



#Restricciones

# Cada cliente es visitado exactamente una vez
for i in N_clientes:
    model += lpSum(x[(j, i)] for j in N if j != i) == 1, f"entrada_{i}"
    model += lpSum(x[(i, j)] for j in N if j != i) == 1, f"salida_{i}"


# Balance de depósito
model += (lpSum(x[(DEPOSITO, j)] for j in N_clientes) == lpSum(x[(i, DEPOSITO)] for i in N_clientes)), "balance_deposito"
 

for i in N_clientes:
    model += (lpSum(x[(i, j)] for j in N if j != i) == lpSum(x[(j, i)] for j in N if j != i)), f"balance_vehiculo_{i}"


#Capacidad
for i in N_clientes:
    model += (lpSum(f[(i, j)] for j in N if j != i) - lpSum(f[(j, i)] for j in N if j != i) == d[i]), f"flujo_carga_{i}"
for i, j in arcos:
    model += f[(i, j)] <= Q * x[(i, j)], f"cap_{i}_{j}"
for j in N_clientes:
    model += f[(DEPOSITO, j)] == 0, f"vacio_salida_{j}"
for i in N_clientes:
    model += f[(i, DEPOSITO)] <= Q * x[(i, DEPOSITO)], f"cap_regreso_{i}"

#-------------------------------------------------------------------------------------------------


# WARM START (Nearest Neighbor + 2-opt)
def nearest_neighbor_vrp(N_clientes, c, d, Q, deposito=DEPOSITO):
    restantes = set(N_clientes)
    rutas = []
    while restantes:
        # arranca cada ruta con el nodo más cercano al depósito que quede
        inicio = min(restantes, key=lambda i: c[deposito][i])
        ruta, carga, actual = [inicio], d[inicio], inicio
        restantes.remove(inicio)
        while True:
            candidatos = [n for n in restantes if carga + d[n] <= Q]
            if not candidatos:
                break
            siguiente = min(candidatos, key=lambda n: c[actual][n])
            ruta.append(siguiente)
            carga += d[siguiente]
            restantes.remove(siguiente)
            actual = siguiente
        rutas.append(ruta)
    return rutas
 
rutas_nn = nearest_neighbor_vrp(N_clientes, c, d, Q)
 
def two_opt(ruta, c, deposito=DEPOSITO):
    seq = [deposito] + ruta + [deposito]
    mejor = seq[:]
    mejorado = True
    while mejorado:
        mejorado = False
        for i in range(1, len(mejor) - 2):
            for j in range(i + 1, len(mejor) - 1):
                nueva = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                costo_actual = sum(c[mejor[k]][mejor[k+1]] for k in range(len(mejor)-1))
                costo_nueva  = sum(c[nueva[k]][nueva[k+1]]  for k in range(len(nueva)-1))
                if costo_nueva < costo_actual:
                    mejor = nueva
                    mejorado = True
    return mejor[1:-1]
 
rutas_nn = [two_opt(r, c) for r in rutas_nn]
 
for v in x.values(): v.setInitialValue(0)
for v in f.values(): v.setInitialValue(0)
 
for ruta in rutas_nn:
    secuencia = [DEPOSITO] + ruta + [DEPOSITO]
    carga_acum = 0
    for k in range(len(secuencia) - 1):
        i, j = secuencia[k], secuencia[k+1]
        x[(i, j)].setInitialValue(1)
        if i == DEPOSITO:
            f[(i, j)].setInitialValue(0)
        else:
            carga_acum += d[i]
            f[(i, j)].setInitialValue(carga_acum)
 
print(f"[NN+2OPT] {len(rutas_nn)} rutas, {sum(len(r) for r in rutas_nn)}/{len(N_clientes)} nodos cubiertos")
for k, ruta in enumerate(rutas_nn):
    print(f"  Ruta {k+1}: {[DEPOSITO]+ruta+[DEPOSITO]}  |  Cajas: {sum(d[i] for i in ruta):.1f}/{Q}")

# -------------------------------
# SOLVER
# -------------------------------
print("\n[INFO] Resolviendo...")
model.solve(PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap, msg=1, warmStart=True, keepFiles=True))

def val(v):
    return value(v) if value(v) is not None else 0.0
 
print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)
print(f"Status:            {LpStatus[model.status]}")
print(f"Distancia total:   {val(model.objective):.4f}")
 
costo_total = sum(c[i][j] * val(x[(i, j)]) for i, j in arcos)
n_rutas     = int(round(sum(val(x[(DEPOSITO, j)]) for j in N_clientes)))
 
print(f"Clientes visitados: {len(N_clientes)} de {len(N_clientes)}")
print(f"Demanda cubierta:   {total_demanda:.1f} de {total_demanda:.1f} cajas")
 
# Reconstruir rutas
arcos_on = {(i, j) for i, j in arcos if val(x[(i, j)]) > 0.5}
print("Arcos utilizados:")
print(arcos_on)

siguiente = {i: j for i, j in arcos_on}
rutas = []
for primer_nodo in [j for j in N_clientes if (DEPOSITO, j) in arcos_on]:
    ruta = []
    actual = primer_nodo
    while actual != DEPOSITO:
        ruta.append(actual)
        actual = siguiente[actual]
    rutas.append(ruta)
# Imprimir rutas
print("\nRUTAS:")
for k, ruta in enumerate(rutas, 1):
    secuencia = [DEPOSITO] + ruta + [DEPOSITO]
    distancia = sum(c[secuencia[i]][secuencia[i+1]]
                    for i in range(len(secuencia)-1))
    carga = sum(d[i] for i in ruta)
    print(
        f"Ruta {k}: {secuencia} | "
        f"Carga: {carga}/{Q} | "
        f"Distancia: {distancia:.2f}"
    )
