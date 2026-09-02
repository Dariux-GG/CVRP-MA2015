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
}     #temp

# Capacidad
Q = 50



#-------------------------------------------------------------------------------------------------
#Modelo
modelo = LpProblem("CVRP", LpMinimize)
#-------------------------------------------------------------------------------------------------

# variables
x = {}
for i in nodos:
    for j in nodos:
        if i != j:
            for k in vehiculos:
                x[i, j, k] = LpVariable(
                    f"x_{i}_{j}_{k}",
                    cat="Binary"
                )
#esto esta interesante pero básicamente si tenemos x_ijk, la variable se guarda como x[i,j,k] 

u = LpVariable.dicts("u", clientes, lowBound=0, cat="Continuous")
#-------------------------------------------------------------------------------------------------


#Restricciones

# Cada cliente es visitado exactamente una vez
for j in clientes:
    modelo += (
        lpSum(
            x[i, j, k]
            for k in vehiculos
            for i in nodos
            if i != j
        ) == 1
    )


# Conservación de flujo
for k in vehiculos:
    for j in nodos:

        entradas = lpSum(
            x[i, j, k]
            for i in nodos
            if i != j
        )

        salidas = lpSum(
            x[j, l, k]
            for l in nodos
            if l != j
        )

        modelo += entradas == salidas


# Cada vuelta sale a lo sumo una vez del depósito
for k in vehiculos:
    modelo += lpSum(
        x[0, j, k]
        for j in clientes
    ) <= 1

# Cada vehículo regresa a lo sumo una vez al depósito
for k in vehiculos:
    modelo += lpSum(
        x[i, 0, k]
        for i in clientes
    ) <= 1


# Capacidad del vehículo
for k in vehiculos:
    modelo += lpSum(
        d[i] * x[i, j, k]
        for i in clientes
        for j in nodos
        if i != j
    ) <= Q


# Prevención de subtours (MTZ)
for k in vehiculos:
    for i in clientes:
        for j in clientes:

            if i != j:

                modelo += (
                    u[i] - u[j]
                    + Q * x[i, j, k]
                    <= Q - d[j]
                )


# Límites de carga acumulada
for i in clientes:

    modelo += u[i] >= d[i]
    modelo += u[i] <= Q


#-------------------------------------------------------------------------------------------------

# objetivo
modelo += lpSum(
    c[i][j] * x[i, j, k]
    for i in nodos
    for j in nodos
    if i != j
    for k in vehiculos
)


#-------------------------------------------------------------------------------------------------
#Solve
#(tiene early stop)
modelo.solve(PULP_CBC_CMD(msg=True, timeLimit=60))


# Resultados

print("RESULTADOS:\n\n")

print("Estado:", LpStatus[modelo.status])
print("Distancia total:", value(modelo.objective))
for k in vehiculos:
    # Arcos utilizados en esta vuelta
    arcos = []
    for i in nodos:
        for j in nodos:
            if i != j:
                if value(x[i, j, k]) == 1:
                    arcos.append((i, j))

    # Si la vuelta no se utiliza
    if len(arcos) == 0:
        continue
    print(f"\nVuelta {k}")
    print("Arcos:", arcos)

    # Construir la ruta
    ruta = [0]
    actual = 0
    while True:
        siguiente = None
        for i, j in arcos:
            if i == actual:
                siguiente = j
                break
        if siguiente is None:
            break
        ruta.append(siguiente)
        if siguiente == 0:
            break
        actual = siguiente
    print("Ruta:", " -> ".join(map(str, ruta)))

    # Distancia de la vuelta
    distancia_vuelta = sum(
        c[i][j]
        for i, j in arcos
    )
    print("Distancia:", distancia_vuelta)

print("\n==============================")