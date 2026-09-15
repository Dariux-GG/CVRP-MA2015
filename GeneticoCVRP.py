import numpy as np
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


def _datos_instancia(instancia):
    if instancia is None:
        return clientes, distancias, nodos_demanda, capacidad_vehiculo, DEPOSITO
    return (instancia.clientes, instancia.distancias, instancia.demandas,
            instancia.capacidad, instancia.deposito)


def generar_poblacion(tamano_poblacion, clientes_instancia=None):
    clientes_actuales = clientes if clientes_instancia is None else clientes_instancia
    poblacion = []
    for _ in range(tamano_poblacion):
        # FIX: permutación completa de los clientes (giant tour), no una
        # muestra de tamaño arbitrario.
        individuo = random.sample(clientes_actuales, len(clientes_actuales))
        poblacion.append(individuo)
    return poblacion


def evaluar_split(individuo, distancias=None, demandas=None,
                   capacidad=None, deposito=None, instancia=None):
    """
    Dado un giant tour (permutación de clientes), encuentra la partición
    óptima en rutas factibles vía programación dinámica.
    Regresa (costo_total, predecesor) donde predecesor permite reconstruir
    los cortes de ruta.
    """
    if instancia is not None:
        _, distancias, demandas, capacidad, deposito = _datos_instancia(instancia)
    else:
        distancias = globals()['distancias'] if distancias is None else distancias
        demandas = nodos_demanda if demandas is None else demandas
        capacidad = (capacidad_vehiculo if capacidad is None else capacidad)
        deposito = DEPOSITO if deposito is None else deposito
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


def fitness(individuo, instancia=None):
    costo, _ = evaluar_split(individuo, instancia=instancia)
    return costo


def seleccion_torneo(poblacion, fitnesses, tam_torneo=3):
    """Elige tam_torneo individuos al azar y regresa el de mejor fitness
    (menor distancia). Se llama una vez por cada padre que necesites."""
    tam_torneo = min(tam_torneo, len(poblacion))
    contendientes_idx = random.sample(range(len(poblacion)), tam_torneo)
    mejor_idx = min(contendientes_idx, key=lambda i: fitnesses[i])
    return poblacion[mejor_idx]


def seleccion_ruleta(poblacion, fitnesses):
    """Selecciona favoreciendo soluciones de menor distancia."""
    pesos = [1 / max(costo, 1e-12) for costo in fitnesses]
    return random.choices(poblacion, weights=pesos, k=1)[0]


def cruce_ox(padre1, padre2):
    """
    Order Crossover (OX): copia un segmento de un padre tal cual, y llena
    las posiciones restantes con los genes del otro padre en su orden
    relativo, saltando los que ya quedaron en el segmento copiado.
    Garantiza que el hijo sea una permutación válida (sin duplicados ni
    clientes faltantes).
    """
    n = len(padre1)
    i, j = sorted(random.sample(range(n), 2))

    def construir_hijo(segmento_de, relleno_de):
        hijo = [None] * n
        hijo[i:j] = segmento_de[i:j]
        usados = set(hijo[i:j])

        pos = j % n
        orden_relleno = relleno_de[j:] + relleno_de[:j]
        for gen in orden_relleno:
            if gen not in usados:
                hijo[pos] = gen
                pos = (pos + 1) % n
        return hijo

    hijo1 = construir_hijo(padre1, padre2)
    hijo2 = construir_hijo(padre2, padre1)
    return hijo1, hijo2


def cruce_un_punto(padre1, padre2):
    """Cruce de un punto que conserva permutaciones válidas."""
    if len(padre1) < 2:
        return padre1[:], padre2[:]
    punto = random.randrange(1, len(padre1))

    def construir_hijo(prefijo, relleno):
        hijo = prefijo[:punto]
        hijo.extend(gen for gen in relleno if gen not in hijo)
        return hijo

    return construir_hijo(padre1, padre2), construir_hijo(padre2, padre1)


def genetico(criterio_paro: int, tamano_poblacion: int, prob_mutacion: float,
             tam_torneo: int = 3, instancia=None, tipo_cruce='dos_puntos',
             metodo_seleccion='torneo', semilla=None, generaciones=None):
    if semilla is not None:
        random.seed(semilla)
    clientes_actuales, _, _, _, _ = _datos_instancia(instancia)
    if tamano_poblacion < 2:
        raise ValueError('tamano_poblacion debe ser al menos 2')
    if not 0 <= prob_mutacion <= 1:
        raise ValueError('prob_mutacion debe estar entre 0 y 1')
    if tipo_cruce not in ('un_punto', 'dos_puntos'):
        raise ValueError('tipo_cruce debe ser un_punto o dos_puntos')
    if metodo_seleccion not in ('torneo', 'ruleta'):
        raise ValueError('metodo_seleccion debe ser torneo o ruleta')

    poblacion = generar_poblacion(tamano_poblacion, clientes_actuales)
    fitnesses = [fitness(individuo, instancia) for individuo in poblacion]
    paro = 0
    generaciones_ejecutadas = 0
    limite_generaciones = generaciones if generaciones is not None else criterio_paro

    while (generaciones_ejecutadas < limite_generaciones and
           (generaciones is not None or paro < criterio_paro)):
        mejoro_generacion = False
        for _ in range(tamano_poblacion):
            # Selección por torneo: cada padre es el ganador de un
            # mini-torneo de tam_torneo individuos elegidos al azar.
            seleccionar = (seleccion_torneo if metodo_seleccion == 'torneo'
                           else seleccion_ruleta)
            padre1 = seleccionar(poblacion, fitnesses, tam_torneo) \
                if metodo_seleccion == 'torneo' else seleccionar(poblacion, fitnesses)
            padre2 = seleccionar(poblacion, fitnesses, tam_torneo) \
                if metodo_seleccion == 'torneo' else seleccionar(poblacion, fitnesses)

            cruce = cruce_un_punto if tipo_cruce == 'un_punto' else cruce_ox
            hijo1, hijo2 = cruce(padre1, padre2)

            if random.random() < prob_mutacion and len(clientes_actuales) >= 2:
                idx1, idx2 = random.sample(range(len(clientes_actuales)), 2)
                hijo1[idx1], hijo1[idx2] = hijo1[idx2], hijo1[idx1]
                hijo2[idx1], hijo2[idx2] = hijo2[idx2], hijo2[idx1]

            fitness_hijo1 = fitness(hijo1, instancia)
            fitness_hijo2 = fitness(hijo2, instancia)
            peor_idx = fitnesses.index(max(fitnesses))

            mejoro = False
            if fitness_hijo1 < fitnesses[peor_idx]:
                poblacion[peor_idx] = hijo1
                fitnesses[peor_idx] = fitness_hijo1
                mejoro = True
                peor_idx = fitnesses.index(max(fitnesses))
            if fitness_hijo2 < fitnesses[peor_idx]:
                poblacion[peor_idx] = hijo2
                fitnesses[peor_idx] = fitness_hijo2
                mejoro = True

            mejoro_generacion = mejoro_generacion or mejoro

        generaciones_ejecutadas += 1
        paro = 0 if mejoro_generacion else paro + 1

    mejor_idx = fitnesses.index(min(fitnesses))
    return poblacion[mejor_idx], fitnesses[mejor_idx], generaciones_ejecutadas

if __name__ == "__main__":
 
    mejor_individuo, mejor_fitness, _ = genetico(
        criterio_paro=3000,
        tamano_poblacion=500,
        prob_mutacion=0.20,
        tam_torneo=4,
    )
 
    print("Mejor individuo (giant tour):", mejor_individuo)
    print("¿Permutación válida?", sorted(mejor_individuo) == sorted(clientes))
    print(f"Fitness (distancia total): {mejor_fitness:.2f}")
 
    costo, predecesor = evaluar_split(mejor_individuo)
    rutas = reconstruir_rutas(mejor_individuo, predecesor)
    print(f"# rutas: {len(rutas)}")
    for r in rutas:
        carga = sum(nodos_demanda[c] for c in r)
        print(f"  ruta {r} -> carga {carga}/{capacidad_vehiculo}")
 