"""
Ejercicio 3: Simulación promedio del modelo SIR con partículas
Con tabla visual de snapshots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

try:
    from scipy.spatial import cKDTree as KDTree
    _KD_AVAILABLE = True
except Exception:
    KDTree = None
    _KD_AVAILABLE = False

# ===== FUNCIONES AUXILIARES =====

def reflect_positions(pos, vel, L):
    """Rebote perfecto en las fronteras"""
    mask_x_low = pos[:, 0] < 0
    if np.any(mask_x_low):
        pos[mask_x_low, 0] = -pos[mask_x_low, 0]
        vel[mask_x_low, 0] = -vel[mask_x_low, 0]
    mask_x_high = pos[:, 0] > L
    if np.any(mask_x_high):
        pos[mask_x_high, 0] = 2 * L - pos[mask_x_high, 0]
        vel[mask_x_high, 0] = -vel[mask_x_high, 0]
    
    mask_y_low = pos[:, 1] < 0
    if np.any(mask_y_low):
        pos[mask_y_low, 1] = -pos[mask_y_low, 1]
        vel[mask_y_low, 1] = -vel[mask_y_low, 1]
    mask_y_high = pos[:, 1] > L
    if np.any(mask_y_high):
        pos[mask_y_high, 1] = 2 * L - pos[mask_y_high, 1]
        vel[mask_y_high, 1] = -vel[mask_y_high, 1]


def step_epidemic(pos, states, beta, gamma, dt, r, use_kdtree=True, rng=None):
    """Actualiza estados S->I y I->R"""
    if rng is None:
        rng = np.random.default_rng()
    
    N = len(states)
    new_states = states.copy()
    
    p_inf = 1.0 - math.exp(-beta * dt)
    p_rec = 1.0 - math.exp(-gamma * dt)
    
    infected_idx = np.where(states == 1)[0]
    susceptible_idx = np.where(states == 0)[0]
    
    if len(infected_idx) == 0:
        return new_states
    
    newly_infected = set()
    
    if use_kdtree and _KD_AVAILABLE and len(susceptible_idx) > 0:
        tree_inf = KDTree(pos[infected_idx])
        tree_sus = KDTree(pos[susceptible_idx])
        neighbors = tree_inf.query_ball_tree(tree_sus, r)
        
        for i_inf, neigh_list in enumerate(neighbors):
            if not neigh_list or p_inf <= 0.0:
                continue
            trials = rng.random(len(neigh_list))
            for j, trial in enumerate(trials):
                if trial < p_inf:
                    sus_global_idx = susceptible_idx[neigh_list[j]]
                    newly_infected.add(int(sus_global_idx))
    else:
        if len(susceptible_idx) > 0:
            INF = pos[infected_idx]
            SUS = pos[susceptible_idx]
            for i, ipos in enumerate(INF):
                d2 = np.sum((SUS - ipos) ** 2, axis=1)
                idxs = np.where(d2 <= r * r)[0]
                if idxs.size == 0 or p_inf <= 0.0:
                    continue
                trials = rng.random(len(idxs))
                for k, trial in enumerate(trials):
                    if trial < p_inf:
                        sus_global_idx = susceptible_idx[idxs[k]]
                        newly_infected.add(int(sus_global_idx))
    
    if newly_infected:
        new_states[list(newly_infected)] = 1
    
    if p_rec > 0.0 and infected_idx.size > 0:
        trials_rec = rng.random(len(infected_idx))
        rec_mask = trials_rec < p_rec
        recovered_idx = infected_idx[rec_mask]
        new_states[recovered_idx] = 2
    
    return new_states


def run_single_simulation(pos_init, vel_init, states_init, L, beta, gamma, dt, T, r, 
                         use_kdtree, seed_sim):
    """Ejecuta una simulación individual"""
    rng = np.random.default_rng(seed_sim)
    pos = pos_init.copy()
    vel = vel_init.copy()
    states = states_init.copy()
    
    n_steps = int(np.ceil(T / dt))
    Ntotal = len(states)
    
    S_t = np.empty(n_steps + 1, dtype=int)
    I_t = np.empty(n_steps + 1, dtype=int)
    R_t = np.empty(n_steps + 1, dtype=int)
    
    S_t[0] = np.count_nonzero(states == 0)
    I_t[0] = np.count_nonzero(states == 1)
    R_t[0] = np.count_nonzero(states == 2)
    
    for step in range(n_steps):
        pos += vel * dt
        reflect_positions(pos, vel, L)
        states = step_epidemic(pos, states, beta, gamma, dt, r, use_kdtree, rng)
        
        S_t[step + 1] = np.count_nonzero(states == 0)
        I_t[step + 1] = np.count_nonzero(states == 1)
        R_t[step + 1] = np.count_nonzero(states == 2)
    
    return S_t, I_t, R_t


def create_snapshot_table(times, S_mean, I_mean, R_mean, snapshot_times, Ntotal, dt):
    """Crea una figura con tabla de snapshots"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos de la tabla
    table_data = []
    table_data.append(['Tiempo (t)', 'S(t)', 'I(t)', 'R(t)', '%S', '%I', '%R', 'Fase'])
    
    for t_snap in snapshot_times:
        idx = int(round(t_snap / dt))
        if 0 <= idx < len(times):
            S_val = S_mean[idx]
            I_val = I_mean[idx]
            R_val = R_mean[idx]
            
            pct_S = 100 * S_val / Ntotal
            pct_I = 100 * I_val / Ntotal
            pct_R = 100 * R_val / Ntotal
            
            # Determinar fase
            if t_snap == 0:
                fase = 'Inicio'
            elif I_val > 0.8 * np.max(I_mean):
                fase = 'Pico'
            elif I_val > 5:
                fase = 'Activa'
            elif I_val > 1:
                fase = 'Descenso'
            else:
                fase = 'Extinción'
            
            table_data.append([
                f'{t_snap:.0f}',
                f'{S_val:.1f}',
                f'{I_val:.1f}',
                f'{R_val:.1f}',
                f'{pct_S:.1f}%',
                f'{pct_I:.1f}%',
                f'{pct_R:.1f}%',
                fase
            ])
    
    # Crear tabla
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.16])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo para el encabezado
    for i in range(8):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Colores alternados para filas
    for i in range(1, len(table_data)):
        for j in range(8):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Snapshots de la Evolución - Modelo de Partículas', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('tabla_snapshots_particulas.png', dpi=150, bbox_inches='tight')
    print("✓ Tabla guardada: tabla_snapshots_particulas.png")
    plt.show()


# ===== FUNCIÓN PRINCIPAL =====

def run_averaged_simulation(L=40.0, Ntotal=100, I0=15, vmax=10.0,
                           r=2.0, beta=0.3, gamma=0.05,
                           dt=0.1, T=150.0, 
                           seed_init=42, Nexp=10,
                           use_kdtree=True,
                           snapshot_times=None):
    """Ejecuta Nexp simulaciones y promedia resultados"""
    
    print(f"\n{'='*60}")
    print(f"EJERCICIO 3: SIMULACIÓN PROMEDIADA - PARTÍCULAS")
    print(f"{'='*60}")
    print(f"Parámetros: L={L}, N={Ntotal}, I0={I0}, vmax={vmax}")
    print(f"           r={r}, β={beta}, γ={gamma}, dt={dt}, T={T}")
    print(f"Repeticiones: Nexp={Nexp}")
    print(f"Semilla inicial: {seed_init}")
    print(f"{'='*60}\n")
    
    # PASO 1: Generar condiciones iniciales FIJAS
    rng_init = np.random.default_rng(seed_init)
    
    pos_init = rng_init.uniform(0.0, L, size=(Ntotal, 2))
    thetas = rng_init.uniform(0.0, 2 * np.pi, size=Ntotal)
    speeds = rng_init.uniform(0.0, vmax, size=Ntotal)
    vel_init = np.vstack((speeds * np.cos(thetas), speeds * np.sin(thetas))).T
    
    states_init = np.zeros(Ntotal, dtype=np.int8)
    if I0 > 0:
        infected_idx = rng_init.choice(Ntotal, size=I0, replace=False)
        states_init[infected_idx] = 1
    
    print(f"✓ Condiciones iniciales generadas")
    
    # PASO 2: Ejecutar Nexp simulaciones
    n_steps = int(np.ceil(T / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    
    all_S = np.zeros((Nexp, n_steps + 1))
    all_I = np.zeros((Nexp, n_steps + 1))
    all_R = np.zeros((Nexp, n_steps + 1))
    
    print(f"\nEjecutando {Nexp} simulaciones...")
    for i in range(Nexp):
        seed_sim = seed_init + 1000 + i
        S_t, I_t, R_t = run_single_simulation(
            pos_init, vel_init, states_init, L, beta, gamma, dt, T, r, 
            use_kdtree, seed_sim
        )
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
    
    print(f"\n✓ Simulaciones completadas\n")
    
    # PASO 4: Visualización principal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_title('Todas las Simulaciones Individuales', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Número de individuos')
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, Ntotal)
    ax1.grid(True, alpha=0.3)
    
    for i in range(Nexp):
        ax1.plot(times, all_S[i], 'b-', alpha=0.2, linewidth=0.5)
        ax1.plot(times, all_I[i], 'r-', alpha=0.2, linewidth=0.5)
        ax1.plot(times, all_R[i], 'g-', alpha=0.2, linewidth=0.5)
    
    ax2.set_title(f'Curvas Promediadas (Nexp={Nexp})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Número de individuos')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, Ntotal)
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
    
    # Marcar snapshots
    if snapshot_times:
        for t_snap in snapshot_times:
            ax2.axvline(x=t_snap, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax2.legend(loc='best', fontsize=8)
    
    param_text = (f'L={L}, N={Ntotal}, I₀={I0}, vmax={vmax}, r={r}, '
                 f'β={beta}, γ={gamma}, dt={dt}, seed={seed_init}')
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('sir_particles_averaged.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfica guardada: sir_particles_averaged.png")
    plt.show()
    
    # PASO 5: Crear tabla de snapshots
    if snapshot_times:
        create_snapshot_table(times, S_mean, I_mean, R_mean, snapshot_times, Ntotal, dt)
    
    # Estadísticas
    I_max_mean = np.max(I_mean)
    t_peak = times[np.argmax(I_mean)]
    final_R_pct = 100.0 * (R_mean[-1] / Ntotal)
    
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS DEL PROMEDIO:")
    print(f"{'='*60}")
    print(f"Pico promedio de infectados: I_max = {I_max_mean:.1f} en t = {t_peak:.1f}")
    print(f"Recuperados finales (promedio): {R_mean[-1]:.1f} ({final_R_pct:.1f}%)")
    print(f"{'='*60}\n")
    
    return {
        't': times,
        'S_mean': S_mean, 'I_mean': I_mean, 'R_mean': R_mean,
        'S_std': S_std, 'I_std': I_std, 'R_std': R_std,
        'all_S': all_S, 'all_I': all_I, 'all_R': all_R
    }


# ===== EJECUCIÓN =====
if __name__ == "__main__":
    params = dict(
        L=40.0,
        Ntotal=100,
        I0=15,
        vmax=10.0,
        r=2.0,
        beta=0.3,
        gamma=0.05,
        dt=0.1,
        T=150.0,
        seed_init=42,
        Nexp=20,
        use_kdtree=True,
        snapshot_times=[0, 20, 40, 60, 90, 120, 150]
    )
    
    results = run_averaged_simulation(**params)