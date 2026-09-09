import numpy as np
import matplotlib.pyplot as plt
import random

nodos = 11
DEPOSITO = 1

nodos_coords = {1: np.array([50, 50]),
                 2: np.array([20, 70]),
                 3: np.array([30, 80]),
                 4: np.array([60, 75]),
                 5: np.array([80, 70]),
                 6: np.array([25, 45]),
                 7: np.array([40, 30]),
                 8: np.array([65, 25]),
                 9: np.array([85, 40]),
                 10: np.array([45, 60]),
                 11: np.array([70, 55])}

nodos_demanda = {1: 0,
                 2: 12,
                 3: 18,
                 4: 10,
                 5: 15,
                 6: 14,
                 7: 20,
                 8: 16,
                 9: 11,
                 10: 13,
                 11: 17}

capacidad_vehiculo = 50

clientes = [n for n in nodos_coords if n != DEPOSITO]  # todos menos el depósito

# --- FIX: matriz de tamaño (nodos+1) para poder indexar directo con el id
# del nodo (1..11) sin tener que restar 1 en todos lados.
distancias = np.zeros((nodos + 1, nodos + 1))
for i in range(1, nodos + 1):
    for j in range(1, nodos + 1):
        distancias[i][j] = np.linalg.norm(nodos_coords[i] - nodos_coords[j])


def evaluar_split(individuo, distancias=distancias, demandas=nodos_demanda,
                   capacidad=capacidad_vehiculo, deposito=DEPOSITO):
    """
    Dado un giant tour (permutación de clientes), encuentra la partición
    óptima en rutas factibles vía programación dinámica.
    Regresa (costo_total, predecesor) donde predecesor permite reconstruir
    los cortes de ruta.
    """
    m = len(individuo)
    costo_min = [float('inf')] * (m + 1)
    predecesor = [None] * (m + 1)
    costo_min[0] = 0

    for i in range(m):
        if costo_min[i] == float('inf'):
            continue
        carga = 0
        dist_ruta = 0
        nodo_anterior = deposito
        for j in range(i, m):
            cliente = individuo[j]
            carga += demandas[cliente]
            if carga > capacidad:
                break
            dist_ruta += distancias[nodo_anterior][cliente]
            nodo_anterior = cliente
            costo_ruta_cerrada = dist_ruta + distancias[cliente][deposito]
            candidato = costo_min[i] + costo_ruta_cerrada
            if candidato < costo_min[j + 1]:
                costo_min[j + 1] = candidato
                predecesor[j + 1] = i

    return costo_min[m], predecesor


def reconstruir_rutas(individuo, predecesor):
    """Convierte el vector de predecesores del Split en la lista de rutas."""
    m = len(individuo)
    rutas = []
    j = m
    while j > 0:
        i = predecesor[j]
        rutas.append(individuo[i:j])
        j = i
    rutas.reverse()
    return rutas


def fitness(individuo):
    costo, _ = evaluar_split(individuo)
    return costo



def construir_hormiga(feromona, alfa, beta):
    """
    Construye un giant tour usando la cantidad de feromona y la distancia
    entre los nodos. Cada hormiga visita todos los clientes una sola vez.
    """
    no_visitados = clientes[:]
    individuo = []
    actual = DEPOSITO

    while no_visitados:

        pesos = []

        # Calculamos qué tan conveniente es ir a cada cliente
        for cliente in no_visitados:

            fer = feromona[actual][cliente] ** alfa

            if distancias[actual][cliente] == 0:
                visibilidad = 0
            else:
                visibilidad = (1 / distancias[actual][cliente]) ** beta

            pesos.append(fer * visibilidad)

        # Elegimos el siguiente cliente usando las probabilidades
        # calculadas con feromona y distancia
        total = sum(pesos)
        probabilidades = [p / total for p in pesos]

        siguiente = random.choices(
            no_visitados,
            weights=probabilidades,
            k=1
        )[0]

        individuo.append(siguiente)
        no_visitados.remove(siguiente)
        actual = siguiente

    return individuo


def actualizar_feromona(feromona, hormigas, fitnesses, rho, Q=100):
    """
    Primero se evapora parte de la feromona y después las hormigas
    agregan feromona en los caminos que utilizaron.
    """
    # Evaporación de feromona
    for i in feromona:
        for j in feromona[i]:
            feromona[i][j] *= (1 - rho)

    # Las mejores soluciones dejan más feromona
    for hormiga, costo in zip(hormigas, fitnesses):

        aporte = Q / costo
        anterior = DEPOSITO

        for cliente in hormiga:

            feromona[anterior][cliente] += aporte
            feromona[cliente][anterior] += aporte

            anterior = cliente


def hormigas_cvrp(criterio_paro, n_hormigas, n_iteraciones,
                  alfa, beta, rho):

    # Al principio todos los caminos tienen la misma cantidad de feromona
    nodos_ids = [DEPOSITO] + clientes

    feromona = {
        i: {
            j: 1.0
            for j in nodos_ids
            if j != i
        }
        for i in nodos_ids
    }

    mejor_individuo = None
    mejor_fitness = float('inf')

    paro = 0
    iteracion = 0

    while paro < criterio_paro and iteracion < n_iteraciones:

        hormigas = []
        fitnesses = []

        # Cada hormiga construye una solución
        for _ in range(n_hormigas):

            individuo = construir_hormiga(
                feromona,
                alfa,
                beta
            )

            costo = fitness(individuo)

            hormigas.append(individuo)
            fitnesses.append(costo)

        # Actualizamos la cantidad de feromona
        actualizar_feromona(
            feromona,
            hormigas,
            fitnesses,
            rho
        )

        # Buscamos la mejor hormiga de esta iteración
        mejor_idx = fitnesses.index(min(fitnesses))

        if fitnesses[mejor_idx] < mejor_fitness:

            mejor_fitness = fitnesses[mejor_idx]
            mejor_individuo = hormigas[mejor_idx]

            # Si encontramos algo mejor, reiniciamos el contador
            paro = 0

        else:
            paro += 1

        iteracion += 1

    return mejor_individuo, mejor_fitness



if __name__ == "__main__":

    random.seed(42)

    mejor_individuo, mejor_fitness = hormigas_cvrp(
        criterio_paro=100,
        n_hormigas=30,
        n_iteraciones=500,
        alfa=1.0,
        beta=3.0,
        rho=0.3,
    )

    print("Mejor individuo (giant tour):", mejor_individuo)
    print("¿Permutación válida?", sorted(mejor_individuo) == sorted(clientes))
    print(f"Fitness (distancia total): {mejor_fitness:.2f}")

    costo, predecesor = evaluar_split(mejor_individuo)
    rutas = reconstruir_rutas(mejor_individuo, predecesor)

    print(f"# rutas: {len(rutas)}")

    for r in rutas:

        carga = sum(nodos_demanda[c] for c in r)

        print(
            f"  ruta {r} -> carga {carga}/{capacidad_vehiculo}"
        )