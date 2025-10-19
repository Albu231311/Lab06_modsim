"""
Ejercicio 3: Simulación promedio del modelo SIR con autómata celular
Con tabla visual de snapshots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

colors = {
    SUSCEPTIBLE: [0.23, 0.51, 0.96],
    INFECTED: [0.94, 0.27, 0.27],
    RECOVERED: [0.06, 0.72, 0.51]
}


def obtener_vecindad(grid, i, j, r):
    """Obtiene vecinos dentro del radio r"""
    M, N = grid.shape
    vecinos = []
    
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < M and 0 <= nj < N:
                vecinos.append(grid[ni, nj])
    
    return vecinos


def actualizar_grid(grid, beta, gamma, r, rng):
    """Actualiza el grid según las reglas SIR"""
    M, N = grid.shape
    nuevo_grid = grid.copy()
    
    for i in range(M):
        for j in range(N):
            celda = grid[i, j]
            
            if celda == SUSCEPTIBLE:
                vecinos = obtener_vecindad(grid, i, j, r)
                if len(vecinos) > 0:
                    infectados_vecinos = sum(1 for v in vecinos if v == INFECTED)
                    prob_infeccion = (infectados_vecinos / len(vecinos)) * beta
                    
                    if rng.random() < prob_infeccion:
                        nuevo_grid[i, j] = INFECTED
            
            elif celda == INFECTED:
                if rng.random() < gamma:
                    nuevo_grid[i, j] = RECOVERED
    
    return nuevo_grid


def contar_estados(grid):
    """Cuenta celdas en cada estado"""
    unique, counts = np.unique(grid, return_counts=True)
    estado_counts = dict(zip(unique, counts))
    
    S = estado_counts.get(SUSCEPTIBLE, 0)
    I = estado_counts.get(INFECTED, 0)
    R = estado_counts.get(RECOVERED, 0)
    
    return S, I, R


def run_single_ca_simulation(grid_init, beta, gamma, r, T, seed_sim):
    """Ejecuta una simulación individual del autómata celular"""
    rng = np.random.default_rng(seed_sim)
    grid = grid_init.copy()
    
    S_t = np.zeros(T + 1, dtype=int)
    I_t = np.zeros(T + 1, dtype=int)
    R_t = np.zeros(T + 1, dtype=int)
    
    S, I, R = contar_estados(grid)
    S_t[0], I_t[0], R_t[0] = S, I, R
    
    for t in range(1, T + 1):
        grid = actualizar_grid(grid, beta, gamma, r, rng)
        S, I, R = contar_estados(grid)
        S_t[t], I_t[t], R_t[t] = S, I, R
    
    return S_t, I_t, R_t


def create_snapshot_table(times, S_mean, I_mean, R_mean, snapshot_times, total_cells):
    """Crea una figura con tabla de snapshots"""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Tiempo (t)', 'S(t)', 'I(t)', 'R(t)', '%S', '%I', '%R', 'Fase'])
    
    for t_snap in snapshot_times:
        if 0 <= t_snap < len(times):
            S_val = S_mean[t_snap]
            I_val = I_mean[t_snap]
            R_val = R_mean[t_snap]
            
            pct_S = 100 * S_val / total_cells
            pct_I = 100 * I_val / total_cells
            pct_R = 100 * R_val / total_cells
            
            if t_snap == 0:
                fase = 'Inicio'
            elif I_val > 0.8 * np.max(I_mean):
                fase = 'Pico'
            elif I_val > 0.5 * np.max(I_mean):
                fase = 'Activa'
            elif I_val > 100:
                fase = 'Descenso'
            else:
                fase = 'Extinción'
            
            table_data.append([
                f'{t_snap}',
                f'{S_val:.0f}',
                f'{I_val:.0f}',
                f'{R_val:.0f}',
                f'{pct_S:.1f}%',
                f'{pct_I:.1f}%',
                f'{pct_R:.1f}%',
                fase
            ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.16])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(8):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(8):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Snapshots de la Evolución - Autómata Celular', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('tabla_snapshots_automata.png', dpi=150, bbox_inches='tight')
    print("✓ Tabla guardada: tabla_snapshots_automata.png")
    plt.show()


def run_averaged_ca_simulation(M=100, N=100, I0=1, beta=0.3, gamma=0.009, 
                               r=2, T=600, seed_init=42, Nexp=10,
                               snapshot_times=None):
    """Ejecuta Nexp simulaciones con las mismas posiciones iniciales"""
    
    print(f"\n{'='*60}")
    print(f"EJERCICIO 3: AUTÓMATA CELULAR PROMEDIADO")
    print(f"{'='*60}")
    print(f"Parámetros: M×N={M}×{N}, I₀={I0}, β={beta}, γ={gamma}, r={r}, T={T}")
    print(f"Repeticiones: Nexp={Nexp}")
    print(f"Semilla inicial: {seed_init}")
    print(f"{'='*60}\n")
    
    # PASO 1: Generar configuración inicial FIJA
    rng_init = np.random.default_rng(seed_init)
    grid_init = np.zeros((M, N), dtype=int)
    
    positions = rng_init.choice(M * N, I0, replace=False)
    infected_positions = []
    for pos in positions:
        i, j = pos // N, pos % N
        grid_init[i, j] = INFECTED
        infected_positions.append((i, j))
    
    print(f"✓ Configuración inicial generada")
    
    # PASO 2: Ejecutar Nexp simulaciones
    all_S = np.zeros((Nexp, T + 1))
    all_I = np.zeros((Nexp, T + 1))
    all_R = np.zeros((Nexp, T + 1))
    
    print(f"\nEjecutando {Nexp} simulaciones...")
    for i in range(Nexp):
        seed_sim = seed_init + 2000 + i
        S_t, I_t, R_t = run_single_ca_simulation(grid_init, beta, gamma, r, T, seed_sim)
        all_S[i] = S_t
        all_I[i] = I_t
        all_R[i] = R_t
        print(f"  Simulación {i+1}/{Nexp} completada")
    
    # PASO 3: Calcular promedios
    S_mean = np.mean(all_S, axis=0)
    I_mean = np.mean(all_I, axis=0)
    R_mean = np.mean(all_R, axis=0)
    
    S_std = np.std(all_S, axis=0)
    I_std = np.std(all_I, axis=0)
    R_std = np.std(all_R, axis=0)
    
    times = np.arange(T + 1)
    total_cells = M * N
    
    print(f"\n✓ Simulaciones completadas\n")
    
    # PASO 4: Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_title('Todas las Simulaciones Individuales', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tiempo (t)')
    ax1.set_ylabel('Número de celdas')
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, total_cells)
    ax1.grid(True, alpha=0.3)
    
    for i in range(Nexp):
        ax1.plot(times, all_S[i], 'b-', alpha=0.2, linewidth=0.5)
        ax1.plot(times, all_I[i], 'r-', alpha=0.2, linewidth=0.5)
        ax1.plot(times, all_R[i], 'g-', alpha=0.2, linewidth=0.5)
    
    ax2.set_title(f'Curvas Promediadas (Nexp={Nexp})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tiempo (t)')
    ax2.set_ylabel('Número de celdas')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, total_cells)
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(times, S_mean, 'b-', linewidth=2, label='S(t) promedio')
    ax2.fill_between(times, S_mean - S_std, S_mean + S_std, 
                     color='blue', alpha=0.2, label='S(t) ± σ')
    
    ax2.plot(times, I_mean, 'r-', linewidth=2, label='I(t) promedio')
    ax2.fill_between(times, I_mean - I_std, I_mean + I_std, 
                     color='red', alpha=0.2, label='I(t) ± σ')
    
    ax2.plot(times, R_mean, 'g-', linewidth=2, label='R(t) promedio')
    ax2.fill_between(times, R_mean - R_std, R_mean + R_std, 
                     color='green', alpha=0.2, label='R(t) ± σ')
    
    if snapshot_times:
        for t_snap in snapshot_times:
            ax2.axvline(x=t_snap, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax2.legend(loc='best', fontsize=8)
    
    param_text = (f'M×N={M}×{N}, I₀={I0}, β={beta}, γ={gamma}, r={r}, seed={seed_init}')
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('sir_cellular_averaged.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfica guardada: sir_cellular_averaged.png")
    plt.show()
    
    # PASO 5: Crear tabla de snapshots
    if snapshot_times:
        create_snapshot_table(times, S_mean, I_mean, R_mean, snapshot_times, total_cells)
    
    # Estadísticas
    I_max_mean = np.max(I_mean)
    t_peak = times[np.argmax(I_mean)]
    final_R_pct = 100.0 * (R_mean[-1] / total_cells)
    
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS DEL PROMEDIO:")
    print(f"{'='*60}")
    print(f"Pico promedio de infectados: I_max = {I_max_mean:.1f} en t = {t_peak}")
    print(f"Recuperados finales (promedio): {R_mean[-1]:.1f} ({final_R_pct:.1f}%)")
    print(f"Susceptibles finales (promedio): {S_mean[-1]:.1f} ({100*S_mean[-1]/total_cells:.1f}%)")
    print(f"{'='*60}\n")
    
    return {
        't': times,
        'S_mean': S_mean, 'I_mean': I_mean, 'R_mean': R_mean,
        'S_std': S_std, 'I_std': I_std, 'R_std': R_std,
        'all_S': all_S, 'all_I': all_I, 'all_R': all_R,
        'grid_init': grid_init
    }


# ===== EJECUCIÓN =====
if __name__ == "__main__":
    params = dict(
        M=100,
        N=100,
        I0=1,
        beta=0.3,
        gamma=0.009,
        r=2,
        T=600,
        seed_init=42,
        Nexp=20,
        snapshot_times=[0, 100, 200, 300, 400, 500, 600]
    )
    
    results = run_averaged_ca_simulation(**params)