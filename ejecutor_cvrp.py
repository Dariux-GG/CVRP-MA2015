"""Ejecutor reproducible para comparar GA y ACO en instancias CVRP."""

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import product
from pathlib import Path
import json
import math
import os
import random
import time

import numpy as np

import GeneticoCVRP as ga
import hormigas as aco


# ========================= CONFIGURACION EDITABLE =========================
ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "Reto_cvrp_primer"
LOG_DIR = ROOT / "hormigas"
INSTANCE_NAMES = []  # Usa [] para ejecutar todas las instancias.
# Tres semillas independientes para repetir cada combinación experimental.
SEEDS = [137, 2849, 9173]
ITERACIONES = 100
POBLACION_GA = 30
HORMIGAS_ACO = 20
CRUCES_GA = ["un_punto", "dos_puntos"]
MUTACIONES_GA = [0.05, 0.20, 0.50]
SELECCIONES_GA = ["torneo", "ruleta"]
ALFAS_ACO = [1.0, 2.0]
BETAS_ACO = [3.0, 5.0]
RHOS_ACO = [0.10, 0.30]
PARAMETROS_ACO = [
    {"alfa": alfa, "beta": beta, "rho": rho}
    for alfa, beta, rho in product(ALFAS_ACO, BETAS_ACO, RHOS_ACO)
]
EJECUTAR_GA = False
EJECUTAR_ACO = True
EJECUTAR_EXACTO = False  # CPLEX queda preparado, pero no se ejecuta por defecto.
INSTANCIAS_EXACTAS = {"CMT1.vrp"}
LIMITE_EXACTO_SEGUNDOS = 90
# El Ryzen 9 9955HX tiene 16 nucleos; se reservan dos para el sistema.
RESERVAR_NUCLEOS = 2
MAX_WORKERS = min(
    14,
    max(1, (os.cpu_count() or 1) - RESERVAR_NUCLEOS),
)
# ===========================================================================


@dataclass
class InstanciaCVRP:
    nombre: str
    capacidad: float
    deposito: int
    coordenadas: dict
    demandas: dict
    distancias: np.ndarray

    @property
    def clientes(self):
        return [nodo for nodo in self.coordenadas if nodo != self.deposito]

    @property
    def dimension(self):
        return len(self.coordenadas)


def _cabecera(linea):
    if ":" not in linea:
        return None, None
    clave, valor = linea.split(":", 1)
    return clave.strip().upper(), valor.strip()


def leer_vrp(ruta):
    """Lee una instancia CVRP con secciones TSPLIB estándar."""
    texto = Path(ruta).read_text(encoding="utf-8")
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    cabeceras = {}
    seccion = None
    coordenadas = {}
    demandas = {}
    deposito = None

    for linea in lineas:
        clave, valor = _cabecera(linea)
        if clave and clave not in {"NODE_COORD_SECTION", "DEMAND_SECTION",
                                   "DEPOT_SECTION", "EOF"}:
            cabeceras[clave] = valor
            continue
        if linea.upper() in {"NODE_COORD_SECTION", "DEMAND_SECTION",
                             "DEPOT_SECTION", "EOF"}:
            seccion = linea.upper()
            if seccion == "EOF":
                break
            continue
        partes = linea.split()
        if seccion == "NODE_COORD_SECTION" and len(partes) >= 3:
            coordenadas[int(partes[0])] = (float(partes[1]), float(partes[2]))
        elif seccion == "DEMAND_SECTION" and len(partes) >= 2:
            demandas[int(partes[0])] = float(partes[1])
        elif seccion == "DEPOT_SECTION":
            nodo = int(partes[0])
            if nodo != -1:
                deposito = nodo

    dimension = int(cabeceras.get("DIMENSION", len(coordenadas)))
    capacidad = float(cabeceras["CAPACITY"])
    if deposito is None:
        raise ValueError(f"{ruta}: falta DEPOT_SECTION")
    if len(coordenadas) != dimension:
        raise ValueError(f"{ruta}: DIMENSION={dimension}, coordenadas={len(coordenadas)}")
    if set(coordenadas) != set(demandas):
        raise ValueError(f"{ruta}: coordenadas y demandas no coinciden")
    if any(demanda > capacidad for demanda in demandas.values()):
        raise ValueError(f"{ruta}: existe una demanda mayor que la capacidad")

    ids = sorted(coordenadas)
    max_id = max(ids)
    distancias = np.zeros((max_id + 1, max_id + 1), dtype=float)
    for origen in ids:
        for destino in ids:
            distancias[origen, destino] = math.dist(
                coordenadas[origen], coordenadas[destino]
            )

    return InstanciaCVRP(
        nombre=Path(ruta).name,
        capacidad=capacidad,
        deposito=deposito,
        coordenadas=coordenadas,
        demandas=demandas,
        distancias=distancias,
    )


def rutas_de_solucion(instancia, individuo):
    costo, predecesor = ga.evaluar_split(individuo, instancia=instancia)
    if not math.isfinite(costo):
        return costo, []
    return costo, ga.reconstruir_rutas(individuo, predecesor)


def validar_solucion(instancia, individuo, rutas):
    esperados = set(instancia.clientes)
    visitados = [cliente for ruta in rutas for cliente in ruta]
    cargas = [sum(instancia.demandas[cliente] for cliente in ruta)
              for ruta in rutas]
    factible = (set(visitados) == esperados and len(visitados) == len(esperados)
                and all(carga <= instancia.capacidad + 1e-9 for carga in cargas))
    return {
        "factible": factible,
        "clientes_servidos": len(visitados),
        "clientes_esperados": len(esperados),
        "rutas": len(rutas),
        "cargas": cargas,
        "violacion_capacidad": max(
            [max(0.0, carga - instancia.capacidad) for carga in cargas] or [0.0]
        ),
    }


def _resultado_base(instancia, algoritmo, semilla, parametros):
    return {
        "instancia": instancia.nombre,
        "algoritmo": algoritmo,
        "semilla": semilla,
        "parametros": parametros,
        "distancia": None,
        "tiempo_segundos": None,
        "iteraciones": 0,
        "factible": False,
        "rutas": 0,
        "clientes_servidos": 0,
        "error": None,
    }


def ejecutar_ga(instancia, semilla, cruce, mutacion, seleccion):
    parametros = {
        "cruce": cruce,
        "prob_mutacion": mutacion,
        "seleccion": seleccion,
        "poblacion": POBLACION_GA,
        "iteraciones_objetivo": ITERACIONES,
    }
    resultado = _resultado_base(instancia, "genetico", semilla, parametros)
    inicio = time.perf_counter()
    try:
        individuo, distancia, iteraciones = ga.genetico(
            criterio_paro=ITERACIONES,
            tamano_poblacion=POBLACION_GA,
            prob_mutacion=mutacion,
            tam_torneo=3,
            instancia=instancia,
            tipo_cruce=cruce,
            metodo_seleccion=seleccion,
            semilla=semilla,
            generaciones=ITERACIONES,
        )
        distancia, rutas = rutas_de_solucion(instancia, individuo)
        resultado.update(validar_solucion(instancia, individuo, rutas))
        resultado["distancia"] = distancia
        resultado["iteraciones"] = iteraciones
    except Exception as error:
        resultado["error"] = f"{type(error).__name__}: {error}"
    resultado["tiempo_segundos"] = time.perf_counter() - inicio
    return resultado


def ejecutar_aco(instancia, semilla, parametros):
    parametros_completos = dict(parametros)
    parametros_completos.update({
        "hormigas": HORMIGAS_ACO,
        "iteraciones_objetivo": ITERACIONES,
    })
    resultado = _resultado_base(instancia, "hormigas", semilla, parametros_completos)
    inicio = time.perf_counter()
    try:
        individuo, distancia, iteraciones = aco.hormigas_cvrp(
            criterio_paro=ITERACIONES,
            n_hormigas=HORMIGAS_ACO,
            n_iteraciones=ITERACIONES,
            instancia=instancia,
            semilla=semilla,
            **parametros,
        )
        distancia, rutas = rutas_de_solucion(instancia, individuo)
        resultado.update(validar_solucion(instancia, individuo, rutas))
        resultado["distancia"] = distancia
        resultado["iteraciones"] = iteraciones
    except Exception as error:
        resultado["error"] = f"{type(error).__name__}: {error}"
    resultado["tiempo_segundos"] = time.perf_counter() - inicio
    return resultado


def resolver_exacto(instancia):
    """Resuelve un CVRP pequeño con docplex; devuelve estado legible si falta CPLEX."""
    resultado = {
        "instancia": instancia.nombre,
        "algoritmo": "docplex_cplex",
        "parametros": {"limite_segundos": LIMITE_EXACTO_SEGUNDOS},
        "distancia": None,
        "tiempo_segundos": None,
        "estado": None,
        "gap": None,
        "error": None,
    }
    inicio = time.perf_counter()
    try:
        from docplex.mp.model import Model

        nodos = [instancia.deposito] + instancia.clientes
        clientes = instancia.clientes
        modelo = Model(name=f"cvrp_{instancia.nombre}")
        arcos = [(i, j) for i in nodos for j in nodos if i != j]
        x = modelo.binary_var_dict(arcos, name="x")
        carga = modelo.continuous_var_dict(clientes, lb=0,
                                           ub=instancia.capacidad, name="carga")
        for cliente in clientes:
            modelo.add_constraint(
                modelo.sum(x[cliente, j] for j in nodos if j != cliente) == 1
            )
            modelo.add_constraint(
                modelo.sum(x[i, cliente] for i in nodos if i != cliente) == 1
            )
            modelo.add_constraint(carga[cliente] >= instancia.demandas[cliente])
        modelo.add_constraint(
            modelo.sum(x[instancia.deposito, j] for j in clientes) ==
            modelo.sum(x[i, instancia.deposito] for i in clientes)
        )
        for i, j in arcos:
            if i != instancia.deposito and j != instancia.deposito:
                modelo.add_constraint(
                    carga[j] >= carga[i] + instancia.demandas[j] -
                    instancia.capacidad * (1 - x[i, j])
                )
        modelo.minimize(modelo.sum(instancia.distancias[i, j] * x[i, j]
                                   for i, j in arcos))
        modelo.parameters.timelimit = LIMITE_EXACTO_SEGUNDOS
        solucion = modelo.solve(log_output=False)
        resultado["estado"] = (str(modelo.solve_details.status)
                                if solucion is not None else "no_solution")
        if solucion is not None:
            resultado["distancia"] = float(solucion.objective_value)
            gap = getattr(modelo.solve_details, "mip_relative_gap", None)
            resultado["gap"] = float(gap) if gap is not None else None
    except Exception as error:
        resultado["error"] = f"{type(error).__name__}: {error}"
    resultado["tiempo_segundos"] = time.perf_counter() - inicio
    return resultado


def _ejecutar_trabajo(trabajo):
    """Ejecuta una tarea independiente dentro de un proceso worker."""
    algoritmo, ruta, parametros = trabajo
    instancia = leer_vrp(ruta)
    if algoritmo == "genetico":
        return ejecutar_ga(instancia, *parametros)
    if algoritmo == "hormigas":
        return ejecutar_aco(instancia, *parametros)
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def instancias_configuradas():
    rutas = sorted(INSTANCE_DIR.glob("*.vrp"))
    if INSTANCE_NAMES:
        permitidas = set(INSTANCE_NAMES)
        rutas = [ruta for ruta in rutas if ruta.name in permitidas]
    if not rutas:
        raise FileNotFoundError(f"No se encontraron instancias en {INSTANCE_DIR}")
    return rutas


def ejecutar():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archivo_log = LOG_DIR / f"ejecucion_{marca_tiempo}.jsonl"
    resultados = []
    trabajos = []
    for ruta in instancias_configuradas():
        instancia = leer_vrp(ruta)
        if EJECUTAR_EXACTO and ruta.name in INSTANCIAS_EXACTAS:
            resultados.append(resolver_exacto(instancia))
        if EJECUTAR_GA:
            for semilla, cruce, mutacion, seleccion in product(
                    SEEDS, CRUCES_GA, MUTACIONES_GA, SELECCIONES_GA):
                trabajos.append((
                    "genetico",
                    str(ruta),
                    (semilla, cruce, mutacion, seleccion),
                ))
        if EJECUTAR_ACO:
            for semilla, parametros in product(SEEDS, PARAMETROS_ACO):
                trabajos.append((
                    "hormigas",
                    str(ruta),
                    (semilla, parametros),
                ))

    if trabajos:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            resultados.extend(executor.map(
                _ejecutar_trabajo,
                trabajos,
                chunksize=1,
            ))

    with archivo_log.open("w", encoding="utf-8") as salida:
        for resultado in resultados:
            salida.write(json.dumps(resultado, ensure_ascii=False) + "\n")
    print(f"Resultados guardados en {archivo_log}")
    print(f"Ejecuciones registradas: {len(resultados)}")
    return resultados


if __name__ == "__main__":
    ejecutar()
