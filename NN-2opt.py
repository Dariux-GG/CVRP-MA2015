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
max_viajes    = math.ceil(total_demanda / Q)
print(f"[INFO] Clientes a rutear: {len(N_clientes)}, demanda total={total_demanda:.1f}")
print(f"[INFO] Q={Q}, max_viajes posibles={max_viajes}")


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
model += (lpSum(x[(DEPOSITO, j)] for j in N_clientes) <= max_viajes), "max_rutas"
 

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
print(f"Rutas abiertas:     {n_rutas}  (máximo permitido={max_viajes})")
 
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

#nada de lo que sigue sirve, es de otro proyecto y no es relevante
"""
# =================================================================
# Animacion
# =================================================================

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

def animar_rutas(rutas, df_coords, deposito='1'):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')

    lons = df_coords["Longitud"].astype(float)
    lats = df_coords["Latitud"].astype(float)

    # todos los nodos de fondo
    ax.scatter(lons, lats, c='#444466', s=20, zorder=2)
    dep_lon = df_coords.loc[deposito, "Longitud"]
    dep_lat = df_coords.loc[deposito, "Latitud"]
    ax.scatter([dep_lon], [dep_lat], c='white', s=120, zorder=5, marker='*')
    ax.set_title("VRP — Rutas óptimas", color='white', fontsize=13)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    colores = plt.cm.tab10.colors
    artistas = []

    def get_coord(nodo):
        return (
            float(df_coords.loc[nodo, "Longitud"]),
            float(df_coords.loc[nodo, "Latitud"])
        )

    # pre-calcular frames: cada frame = un arco de una ruta
    frames_data = []  # (ruta_idx, secuencia hasta ese arco)
    for r_idx, ruta in enumerate(rutas):
        seq = [deposito] + ruta + [deposito]
        for k in range(1, len(seq)):
            frames_data.append((r_idx, seq[:k+1]))

    def update(frame):
        for a in artistas:
            a.remove()
        artistas.clear()

        r_idx, seq_parcial = frames_data[frame]
        color = colores[r_idx % len(colores)]

        # dibujar todas las rutas anteriores completas
        for prev_r in range(r_idx):
            prev_seq = [deposito] + rutas[prev_r] + [deposito]
            prev_color = colores[prev_r % len(colores)]
            for k in range(len(prev_seq)-1):
                x0, y0 = get_coord(prev_seq[k])
                x1, y1 = get_coord(prev_seq[k+1])
                ln, = ax.plot([x0, x1], [y0, y1], color=prev_color, lw=1.5, alpha=0.6, zorder=3)
                artistas.append(ln)

        # dibujar ruta actual hasta el arco actual
        for k in range(len(seq_parcial)-1):
            x0, y0 = get_coord(seq_parcial[k])
            x1, y1 = get_coord(seq_parcial[k+1])
            ln, = ax.plot([x0, x1], [y0, y1], color=color, lw=2.2, alpha=0.95, zorder=4)
            artistas.append(ln)

        # nodos visitados en esta ruta hasta ahora
        for nodo in seq_parcial[1:]:
            if nodo != deposito:
                cx, cy = get_coord(nodo)
                sc = ax.scatter([cx], [cy], c=[color], s=45, zorder=5)
                artistas.append(sc)

        # label de ruta actual
        lbl = ax.text(
            0.01, 0.98, f"Ruta {r_idx+1} / {len(rutas)}",
            transform=ax.transAxes, color=color,
            fontsize=11, va='top', fontweight='bold'
        )
        artistas.append(lbl)

        return artistas

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames_data),
        interval=120,       # ms por frame — sube si va muy rápido
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    return ani

# ---------------------------------------------
# GIF 2 — Satisfacción de demanda por nodo

def animar_demanda(rutas, df_coords, d, deposito='1'):
    """
    #Nodos con demanda pendiente: punto rojo + número de cajas en texto pequeño.
    #Al ser visitado: texto desaparece, punto se vuelve verde.
"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
 
    lons = df_coords["Longitud"].astype(float)
    lats = df_coords["Latitud"].astype(float)
 
    todos = [n for n in df_coords.index if n != deposito and n in d and d[n] > 0]
    total_demanda = sum(d[i] for i in todos)
 
    # Nodos sin demanda al fondo
    nodos_fondo = [n for n in df_coords.index if n not in todos and n != deposito]
    if nodos_fondo:
        fx = [float(df_coords.loc[n, "Longitud"]) for n in nodos_fondo]
        fy = [float(df_coords.loc[n, "Latitud"])  for n in nodos_fondo]
        ax.scatter(fx, fy, c='#2a2a44', s=12, zorder=1)
 
    dep_lon = float(df_coords.loc[deposito, "Longitud"])
    dep_lat = float(df_coords.loc[deposito, "Latitud"])
    ax.scatter([dep_lon], [dep_lat], c='white', s=160, zorder=6, marker='*')
 
    colores_ruta = plt.cm.tab10.colors
 
    # Offset en grados para el texto (pequeño, justo encima del punto)
    # Se ajusta automáticamente al rango del mapa
    lat_range = lats.max() - lats.min()
    TEXT_OFFSET = lat_range * 0.012   # ~1.2% del rango total
 
    def get_coord(nodo):
        return (
            float(df_coords.loc[nodo, "Longitud"]),
            float(df_coords.loc[nodo, "Latitud"])
        )
 
    # Construir secuencia de frames
    frames_events = []
    satisfechos   = set()
    satisfecha    = 0.0
 
    for r_idx, ruta in enumerate(rutas):
        seq = [deposito] + ruta + [deposito]
        for k in range(len(seq)):
            nodo = seq[k]
            if nodo != deposito and nodo not in satisfechos:
                satisfechos.add(nodo)
                satisfecha += d[nodo]
            frames_events.append({
                'ruta_idx'   : r_idx,
                'seq_hasta'  : seq[:k+1],
                'satisfechos': satisfechos.copy(),
                'satisfecha' : satisfecha,
            })
 
    artistas = []
 
    def update(frame):
        for a in artistas:
            a.remove()
        artistas.clear()
 
        ev        = frames_events[frame]
        sat_set   = ev['satisfechos']
        seq_p     = ev['seq_hasta']
        r_idx     = ev['ruta_idx']
        sat_total = ev['satisfecha']
 
        # Arcos rutas anteriores
        for prev_r in range(r_idx):
            prev_seq   = [deposito] + rutas[prev_r] + [deposito]
            prev_color = colores_ruta[prev_r % len(colores_ruta)]
            for k in range(len(prev_seq)-1):
                x0, y0 = get_coord(prev_seq[k])
                x1, y1 = get_coord(prev_seq[k+1])
                ln, = ax.plot([x0,x1],[y0,y1], color=prev_color, lw=1.4, alpha=0.5, zorder=3)
                artistas.append(ln)
 
        # Arcos ruta actual
        cur_color = colores_ruta[r_idx % len(colores_ruta)]
        for k in range(len(seq_p)-1):
            x0, y0 = get_coord(seq_p[k])
            x1, y1 = get_coord(seq_p[k+1])
            ln, = ax.plot([x0,x1],[y0,y1], color=cur_color, lw=2.0, alpha=0.9, zorder=4)
            artistas.append(ln)
 
        # Nodos clientes
        for nodo in todos:
            cx, cy = get_coord(nodo)
 
            if nodo in sat_set:
                # Atendido: verde, sin etiqueta
                sc = ax.scatter([cx], [cy], c='#00e676', s=40, zorder=5)
                artistas.append(sc)
            else:
                # Pendiente: rojo + texto con número de cajas
                sc = ax.scatter([cx], [cy], c='#ff3333', s=35, zorder=5)
                artistas.append(sc)
                txt = ax.text(
                    cx, cy + TEXT_OFFSET,
                    str(int(d[nodo])),
                    color='white', fontsize=5.5,
                    ha='center', va='bottom',
                    zorder=6, fontweight='bold'
                )
                artistas.append(txt)
 
        # Contador global
        pct = (sat_total / total_demanda * 100) if total_demanda > 0 else 0
        titulo = ax.text(
            0.5, 1.005,
            f"Demanda satisfecha: {int(sat_total)} / {int(total_demanda)} cajas  ({pct:.1f}%)"
            f"   — Ruta {r_idx+1}/{len(rutas)}",
            transform=ax.transAxes, color='white', fontsize=10,
            ha='center', va='bottom', fontweight='bold'
        )
        artistas.append(titulo)
 
        # Leyenda
        leg_items = [
            mpatches.Patch(color='#ff3333', label='Demanda pendiente'),
            mpatches.Patch(color='#00e676', label='Demanda satisfecha'),
        ]
        leg = ax.legend(handles=leg_items, loc='lower left',
                        facecolor='#1a1a2e', edgecolor='#444466',
                        labelcolor='white', fontsize=9)
        artistas.append(leg)
 
        return artistas
 
    ani = animation.FuncAnimation(
        fig, update,
        frames=list(range(len(frames_events)))+[len(frames_events)-1]*8,
        interval=200,
        blit=False,
        repeat=False
    )
    plt.tight_layout()
    return ani

    
# -------------------------------
# LLAMADA
# -------------------------------
if salidas:
    rutas_finales = []
    for inicio in salidas:
        ruta = [inicio]
        actual = inicio
        visitados_ruta = {'1', inicio}
        for _ in range(len(N)):
            sig = next(
                (j for j in N if (actual, j) in arcos_on and j not in visitados_ruta),
                None
            )
            if sig is None:
                break
            ruta.append(sig)
            visitados_ruta.add(sig)
            actual = sig
            if actual == '1':
                break
        if ruta and ruta[-1] == '1':
            ruta = ruta[:-1]
        rutas_finales.append(ruta)
 
    # GIF 1
    ani1 = animar_rutas(rutas_finales, df)
    print("[INFO] Guardando rutas_vrp.gif ...")
    ani1.save("rutas_vrp.gif", writer="pillow", fps=8, dpi=120)
    print("[INFO] rutas_vrp.gif guardado.")
    plt.close('all')
 
    # GIF 2
    ani2 = animar_demanda(rutas_finales, df, d)
    print("[INFO] Guardando demanda_vrp.gif ...")
    ani2.save("demanda_vrp.gif", writer="pillow", fps=5, dpi=120)
    print("[INFO] demanda_vrp.gif guardado.")
    plt.close('all')
    """