"""
Simulación de partículas del modelo SIR (Kermack–McKendrick) en [0,L]x[0,L].

Entrega:
 - Código único, reproducible y comentado.
 - Visualización interactiva con matplotlib.animation.FuncAnimation.
 - Guardado opcional de snapshots PNG y datos numéricos (.npz o .csv).
 - Alternativa eficiente usando scipy.spatial.cKDTree si está disponible.
 - No genera .gif ni .mp4.

Lectura rápida (resumen en español, <=10 líneas):
 Este script simula N partículas que se mueven rectilíneamente en un cuadrado
 con rebotes perfectos en los bordes. Infectados transmiten a susceptibles
 dentro de radio r con probabilidad por paso p_inf = 1 - exp(-beta*dt).
 Infectados se recuperan con probabilidad p_rec = 1 - exp(-gamma*dt) por paso.
 Las probabilidades provienen de discretizar procesos de Poisson (p = 1 - exp(-rate*dt)).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

# Tratamiento opcional de KDTree (más eficiente para vecinos)
try:
    from scipy.spatial import cKDTree as KDTree  # rápido y recomendado para N grande
    _KD_AVAILABLE = True
except Exception:
    KDTree = None
    _KD_AVAILABLE = False

# -----------------------------
# Funciones auxiliares principales
# -----------------------------

def check_params(L, Ntotal, I0, vmax, r, beta, gamma, dt, T, seed):
    """Verificaciones básicas de parámetros; lanza ValueError con mensaje claro si algo falla."""
    if L <= 0:
        raise ValueError("L debe ser > 0.")
    if Ntotal <= 0 or not isinstance(Ntotal, int):
        raise ValueError("Ntotal debe ser un entero positivo.")
    if I0 < 0 or I0 > Ntotal or not isinstance(I0, int):
        raise ValueError("I0 debe ser entero en [0, Ntotal].")
    if vmax < 0:
        raise ValueError("vmax debe ser >= 0.")
    if r < 0:
        raise ValueError("r debe ser >= 0.")
    if beta < 0 or gamma < 0:
        raise ValueError("beta y gamma deben ser >= 0.")
    if dt <= 0:
        raise ValueError("dt debe ser > 0.")
    if T <= 0:
        raise ValueError("T debe ser > 0.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed debe ser un entero para reproducibilidad.")


def initialize(L, Ntotal, I0, vmax, seed=0):
    """
    Inicializa posiciones, velocidades y estados.
    Estados: 0 = susceptible, 1 = infectado, 2 = recuperado
    """
    rng = np.random.default_rng(seed)
    # Posiciones uniformes en [0, L]
    pos = rng.uniform(0.0, L, size=(Ntotal, 2))

    # Direcciones angulares uniformes [0, 2pi), magnitudes uniformes en [0, vmax]
    thetas = rng.uniform(0.0, 2 * np.pi, size=Ntotal)
    speeds = rng.uniform(0.0, vmax, size=Ntotal)
    vel = np.vstack((speeds * np.cos(thetas), speeds * np.sin(thetas))).T

    # Estados: elegir I0 índices infectados al azar
    states = np.zeros(Ntotal, dtype=np.int8)  # todos susceptibles inicialmente
    if I0 > 0:
        infected_idx = rng.choice(Ntotal, size=I0, replace=False)
        states[infected_idx] = 1

    return pos, vel, states


def reflect_positions(pos, vel, L):
    """
    Aplica condiciones de frontera de rebote perfecto (reflexión).
    Actualiza posiciones y velocidades en sitio.
    L: tamaño del cuadrado (0..L)
    Reglas:
      - Si x < 0: x = -x, vx = -vx
      - Si x > L: x = 2L - x, vx = -vx
    Se aplica por componente.
    """
    # Eje x (col 0)
    mask_x_low = pos[:, 0] < 0
    if np.any(mask_x_low):
        pos[mask_x_low, 0] = -pos[mask_x_low, 0]
        vel[mask_x_low, 0] = -vel[mask_x_low, 0]
    mask_x_high = pos[:, 0] > L
    if np.any(mask_x_high):
        pos[mask_x_high, 0] = 2 * L - pos[mask_x_high, 0]
        vel[mask_x_high, 0] = -vel[mask_x_high, 0]

    # Eje y (col 1)
    mask_y_low = pos[:, 1] < 0
    if np.any(mask_y_low):
        pos[mask_y_low, 1] = -pos[mask_y_low, 1]
        vel[mask_y_low, 1] = -vel[mask_y_low, 1]
    mask_y_high = pos[:, 1] > L
    if np.any(mask_y_high):
        pos[mask_y_high, 1] = 2 * L - pos[mask_y_high, 1]
        vel[mask_y_high, 1] = -vel[mask_y_high, 1]

    # Nota: una partícula que "sobrepase" el borde mucho se reflecta potencialmente varias veces
    # en una sola actualización; con dt pequeño/vmax razonable esto no es un problema.


def step_epidemic(pos, states, beta, gamma, dt, r,
                  use_kdtree=True, rng=None):
    """
    Actualiza estados S->I y I->R para un paso dt.
    - pos: (N,2) posiciones
    - states: array (N,) con valores 0,1,2
    - beta, gamma: tasas del modelo continuo SIR
    - dt: paso temporal
    - r: radio de contagio
    - use_kdtree: si True y scipy disponible, usa cKDTree para acelerar búsquedas
    - rng: np.random.Generator para reproducibilidad
    Devuelve: new_states (modificado en sitio también)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(states)
    new_states = states  # modificamos en sitio

    # Probabilidades por paso (discretización de Poisson)
    p_inf = 1.0 - math.exp(-beta * dt)  # probabilidad por paso de infectar (por contacto)
    p_rec = 1.0 - math.exp(-gamma * dt)  # probabilidad por paso de recuperación

    # Indices
    infected_idx = np.where(states == 1)[0]
    susceptible_idx = np.where(states == 0)[0]

    if len(infected_idx) == 0:
        # No infectados: solo posible recuperaciones (ninguna)
        return new_states

    if len(susceptible_idx) == 0:
        # No susceptibles: solo posibles recuperaciones
        # Procesar recuperaciones abajo
        pass

    # --- Contagio ---
    # Dos implementaciones:
    # 1) Si KDTree disponible y use_kdtree=True: usar query_ball_tree (eficiente).
    # 2) Si no, hacer método O(N_inf * N_sus) con broadcasting parcial o completo
    newly_infected = set()

    if use_kdtree and _KD_AVAILABLE:
        # Construimos árboles separados para infectados y susceptibles
        tree_inf = KDTree(pos[infected_idx])
        tree_sus = KDTree(pos[susceptible_idx])
        # Query: para cada infectado, lista de índices en susceptibles dentro de r
        # resultado: lista (len(infected_idx)) de listas
        neighbors = tree_inf.query_ball_tree(tree_sus, r)
        # neighbors[i] contiene índices (relativos a susceptible_idx) que están dentro de r
        for i_inf, neigh_list in enumerate(neighbors):
            if not neigh_list:
                continue
            # Para cada susceptible candidato probamos infección con p_inf
            # Generamos una muestra aleatoria para cada vecino
            if p_inf <= 0.0:
                continue
            trials = rng.random(len(neigh_list))
            for j, trial in enumerate(trials):
                if trial < p_inf:
                    sus_global_idx = susceptible_idx[neigh_list[j]]
                    newly_infected.add(int(sus_global_idx))
        # fin KDTree contagio
    else:
        # O(N^2) naive (vectorizado parcialmente).
        # AVISO: coste O(N_inf * N_sus) - puede ser muy lento si N grande.
        # Implementamos de forma que evite la construcción de la matriz completa cuando posible.
        if len(infected_idx) * len(susceptible_idx) > 0:
            # Haremos por bloques si es demasiado grande para memoria.
            # Pero aquí implementamos una versión directa para claridad.
            # (Si quieres velocidad en N grande, instala scipy y activa KDTree)
            INF = pos[infected_idx]  # (Ninf,2)
            SUS = pos[susceptible_idx]  # (Ns,2)
            # Compute pairwise distances efficiently using broadcasting if pequeño:
            # Cuidado con O(N^2) memoria. Chequear tamaño
            max_pairs = 5_000_000  # umbral para evitar crear matrices gigantes
            if INF.shape[0] * SUS.shape[0] <= max_pairs:
                # distancia matriz
                d2 = np.sum((INF[:, None, :] - SUS[None, :, :]) ** 2, axis=2)
                within = d2 <= r * r
                # for each pair within distance, sample infection
                if p_inf > 0.0:
                    inf_pairs = np.argwhere(within)
                    # inf_pairs: rows (i_inf, j_sus)
                    if inf_pairs.size > 0:
                        # Por cada susceptible que tiene al menos un infectado a distancia,
                        # aplicamos un único ensayo por infectado-susceptible.
                        # Esto puede infectar la misma susceptible varias veces; set evita duplicados.
                        trials = rng.random(len(inf_pairs))
                        for k, (i_inf, j_sus) in enumerate(inf_pairs):
                            if trials[k] < p_inf:
                                sus_global_idx = susceptible_idx[j_sus]
                                newly_infected.add(int(sus_global_idx))
            else:
                # Bloqueamos sobre infectados para no saturar memoria
                for i, ipos in enumerate(INF):
                    # dist squared a todos los susceptibles
                    d2 = np.sum((SUS - ipos) ** 2, axis=1)
                    idxs = np.where(d2 <= r * r)[0]
                    if idxs.size == 0 or p_inf <= 0.0:
                        continue
                    trials = rng.random(len(idxs))
                    for k, trial in enumerate(trials):
                        if trial < p_inf:
                            sus_global_idx = susceptible_idx[idxs[k]]
                            newly_infected.add(int(sus_global_idx))

    # Aplicar nuevas infecciones
    if newly_infected:
        new_states[list(newly_infected)] = 1

    # --- Recuperaciones ---
    if p_rec > 0.0 and infected_idx.size > 0:
        trials_rec = rng.random(len(infected_idx))
        rec_mask = trials_rec < p_rec
        recovered_idx = infected_idx[rec_mask]
        new_states[recovered_idx] = 2

    # Asegurar conservación (S+I+R = N)
    # (No cambio en N, solo en estados)
    return new_states


# -----------------------------
# Función principal de simulación
# -----------------------------
def run_simulation(L=100.0, Ntotal=500, I0=5, vmax=1.0,
                   r=1.5, beta=0.3, gamma=0.1,
                   dt=0.1, T=200.0, seed=42,
                   use_kdtree=True,
                   save_data=False, data_filename="sir_particles_data.npz",
                   snapshot_times=None, snapshot_folder="snapshots",
                   show_animation=True, animation_interval=50):
    """
    Ejecuta la simulación SIR por partículas y presenta visualización.
    Parámetros (todos fácilmente modificables):
      - L: tamaño del cuadrado
      - Ntotal: número de partículas
      - I0: infectados iniciales
      - vmax: velocidad máxima
      - r: radio de contagio
      - beta: tasa de infección (continua)
      - gamma: tasa de recuperación (continua)
      - dt: paso temporal
      - T: tiempo total
      - seed: semilla RNG
      - use_kdtree: usar KDTree si está disponible (recomendado)
      - save_data: si True guarda S,I,R,t en data_filename (.npz)
      - snapshot_times: lista de tiempos en los cuales se guardan imágenes PNG
      - snapshot_folder: carpeta para snapshots
      - show_animation: si True crea FuncAnimation para visualización interactiva
      - animation_interval: ms entre frames en la animación (visual only)
    Devuelve: diccionario con arrays y metadatos.
    """
    check_params(L, Ntotal, I0, vmax, r, beta, gamma, dt, T, seed)

    rng = np.random.default_rng(seed)

    # Inicializar
    pos, vel, states = initialize(L, Ntotal, I0, vmax, seed=seed)
    n_steps = int(np.ceil(T / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # Registro S, I, R
    S_t = np.empty(n_steps + 1, dtype=int)
    I_t = np.empty(n_steps + 1, dtype=int)
    R_t = np.empty(n_steps + 1, dtype=int)

    def record(t_idx):
        S_t[t_idx] = np.count_nonzero(states == 0)
        I_t[t_idx] = np.count_nonzero(states == 1)
        R_t[t_idx] = np.count_nonzero(states == 2)

    record(0)

    # Preparar snapshots
    if snapshot_times is None:
        snapshot_times = []
    # Convertir snapshot_times a índices de pasos
    snapshot_indices = set()
    for ts in snapshot_times:
        if ts < 0 or ts > n_steps * dt:
            continue
        idx = int(round(ts / dt))
        snapshot_indices.add(idx)
    if snapshot_indices and not os.path.isdir(snapshot_folder):
        os.makedirs(snapshot_folder, exist_ok=True)

    # Visualización: preparar figura
    fig, (ax_scatter, ax_time) = plt.subplots(1, 2, figsize=(12, 6))
    plt.tight_layout()

    # Scatter inicial
    cmap = {0: "blue", 1: "red", 2: "green"}
    colors = np.array([cmap[s] for s in states])
    scatter = ax_scatter.scatter(pos[:, 0], pos[:, 1], c=colors, s=10, alpha=0.8)
    ax_scatter.set_xlim(0, L)
    ax_scatter.set_ylim(0, L)
    ax_scatter.set_title("Partículas (SIR)")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")

    # Curvas SIR en el tiempo
    line_S, = ax_time.plot(times[:1], S_t[:1], label="S(t)", lw=1)
    line_I, = ax_time.plot(times[:1], I_t[:1], label="I(t)", lw=1)
    line_R, = ax_time.plot(times[:1], R_t[:1], label="R(t)", lw=1)
    ax_time.set_xlim(0, n_steps * dt)
    ax_time.set_ylim(0, Ntotal)
    ax_time.set_xlabel("Tiempo")
    ax_time.set_ylabel("Número de individuos")
    ax_time.set_title("S(t), I(t), R(t)")
    ax_time.legend()

    # Añadir anotación con parámetros
    param_text = (
        f"L={L}, N={Ntotal}, I0={I0}, vmax={vmax}\n"
        f"r={r}, beta={beta}, gamma={gamma}, dt={dt}"
    )
    ax_scatter.text(0.01, 1.02, param_text, transform=ax_scatter.transAxes, fontsize=8,
                    verticalalignment='bottom')

    # Función de actualización para animación
    # NOTA: la animación actualiza en pasos de simulación; para visual en notebook ajusta interval
    def update(frame):
        nonlocal pos, vel, states
        # Movimiento
        pos += vel * dt
        reflect_positions(pos, vel, L)

        # Epidemia (cambiar estados)
        states = step_epidemic(pos, states, beta, gamma, dt, r,
                               use_kdtree=use_kdtree, rng=rng)

        # Registrar
        t_idx = frame + 1  # frame 0 -> paso 1
        record(t_idx)

        # Actualizar scatter y curvas
        colors = np.array([cmap[s] for s in states])
        scatter.set_offsets(pos)
        scatter.set_color(colors)

        # Actualizar curvas parciales
        line_S.set_data(times[:t_idx + 1], S_t[:t_idx + 1])
        line_I.set_data(times[:t_idx + 1], I_t[:t_idx + 1])
        line_R.set_data(times[:t_idx + 1], R_t[:t_idx + 1])
        # Ajustar límites en y si necesario (opcional)
        ax_time.set_ylim(0, max(10, int(Ntotal * 1.05)))

        # Snapshots
        if t_idx in snapshot_indices:
            fname = os.path.join(snapshot_folder, f"snapshot_t{t_idx*dt:.3f}.png")
            fig.savefig(fname, dpi=150)
            print(f"Snapshot guardado: {fname}")

        return scatter, line_S, line_I, line_R

    # Run loop (si show_animation True, use FuncAnimation; si no, correr en bucle sin animación)
    if show_animation:
        # Crear animación (no la guardamos a archivo)
        anim = FuncAnimation(fig, update, frames=n_steps, interval=animation_interval, blit=False)
        plt.show()
    else:
        # Simulación sin animación: ejecutamos todos los pasos y actualizamos datos
        for frame in range(n_steps):
            update(frame)

    # Cálculos finales de salida
    t_array = times
    # Guardar datos si se solicita
    if save_data:
        # Guardar en formato .npz (numpy) y .csv (para compatibilidad)
        np.savez(data_filename, t=t_array, S=S_t, I=I_t, R=R_t)
        # CSV
        csv_fname = os.path.splitext(data_filename)[0] + ".csv"
        header = "t,S,I,R"
        data_stack = np.vstack((t_array, S_t, I_t, R_t)).T
        np.savetxt(csv_fname, data_stack, delimiter=",", header=header, comments='')
        print(f"Datos guardados en {data_filename} y {csv_fname}")

    # Estadísticas imprimibles
    I_max = np.max(I_t)
    t_peak = t_array[np.argmax(I_t)]
    final_recovered_pct = 100.0 * (R_t[-1] / Ntotal)

    print(f"Peak infectados I_max = {I_max} en t = {t_peak:.3f}")
    print(f"Porcentaje final recuperado = {final_recovered_pct:.2f}%")

    # Asserts rápidos para verificar conservación de N
    assert np.all(S_t + I_t + R_t == Ntotal), "Conservación de N violada: S+I+R debe ser siempre Ntotal"

    results = dict(
        t=t_array,
        S=S_t.copy(),
        I=I_t.copy(),
        R=R_t.copy(),
        pos=pos.copy(),
        vel=vel.copy(),
        states=states.copy(),
        params=dict(L=L, Ntotal=Ntotal, I0=I0, vmax=vmax, r=r,
                    beta=beta, gamma=gamma, dt=dt, T=T, seed=seed),
        summary=dict(I_max=int(I_max), t_peak=float(t_peak), final_recovered_pct=float(final_recovered_pct))
    )
    return results


# -----------------------------
# Ejemplo de ejecución (puedes modificar parámetros)
# -----------------------------
if __name__ == "__main__":
    # Parámetros de ejemplo sugeridos
    example_params = dict(
        L=40.0,
        Ntotal=100,
        I0=15,
        vmax=10.0,
        r=2,
        beta=0.3,
        gamma=0.05,
        dt=0.1,
        T=150.0,
        seed=42,
        use_kdtree=True,       # si scipy está instalado será usado
        save_data=False,       # True para guardar arrays .npz y .csv
        snapshot_times=[0.0, 10.0, 50.0, 100.0, 200.0],  # tiempos en los que tomar snapshots (opcional)
        snapshot_folder="snapshots",
        show_animation=True,   # Cambiar a False si no quieres la animación interactiva
        animation_interval=30  # ms por frame para la visualización
    )

    print("KDTree disponible:", _KD_AVAILABLE)
    results = run_simulation(**example_params)

    # Resultado resumido esperado (salida impresa en consola):
    #   - Indicará I_max, t_peak y porcentaje final recuperado.
    #   - Las curvas S(t), I(t), R(t) mostrarán comportamiento SIR: I(t) sube hasta pico y luego baja.
    #   - Con los parámetros de ejemplo: beta=0.3/gamma=0.1 (R0≈3 si contacto efectivo) se espera un brote
    #     con un pico significativo de infectados y una fracción final recuperada alta.