import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

# Parámetros del modelo
M, N = 100, 100  # Tamaño del grid
I0 = 1          # Número de infectados iniciales
beta = 0.3       # Tasa de infección
gamma = 0.009     # Tasa de recuperación
r = 2           # Radio de vecindad
T = 600          # Pasos de tiempo totales

# Estados: 0=Susceptible, 1=Infectado, 2=Recuperado
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

# Colores para visualización
colors = {
    SUSCEPTIBLE: [0.23, 0.51, 0.96],  # Azul
    INFECTED: [0.94, 0.27, 0.27],      # Rojo
    RECOVERED: [0.06, 0.72, 0.51]      # Verde
}

def inicializar_grid(M, N, I0):
    """Inicializa el grid con I0 infectados aleatorios"""
    grid = np.zeros((M, N), dtype=int)
    
    # Colocar I0 infectados en posiciones aleatorias
    positions = np.random.choice(M * N, I0, replace=False)
    for pos in positions:
        i, j = pos // N, pos % N
        grid[i, j] = INFECTED
    
    return grid

def obtener_vecindad(grid, i, j, r):
    """Obtiene los vecinos dentro del radio r"""
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

def actualizar_grid(grid, beta, gamma, r):
    """Actualiza el grid según las reglas del modelo SIR"""
    M, N = grid.shape
    nuevo_grid = grid.copy()
    
    for i in range(M):
        for j in range(N):
            celda = grid[i, j]
            
            if celda == SUSCEPTIBLE:
                # Calcular probabilidad de infección
                vecinos = obtener_vecindad(grid, i, j, r)
                if len(vecinos) > 0:
                    infectados_vecinos = sum(1 for v in vecinos if v == INFECTED)
                    prob_infeccion = (infectados_vecinos / len(vecinos)) * beta
                    
                    if np.random.random() < prob_infeccion:
                        nuevo_grid[i, j] = INFECTED
            
            elif celda == INFECTED:
                # Recuperación con probabilidad gamma
                if np.random.random() < gamma:
                    nuevo_grid[i, j] = RECOVERED
            
            # Los recuperados permanecen recuperados
    
    return nuevo_grid

def contar_estados(grid):
    """Cuenta el número de celdas en cada estado"""
    unique, counts = np.unique(grid, return_counts=True)
    estado_counts = dict(zip(unique, counts))
    
    S = estado_counts.get(SUSCEPTIBLE, 0)
    I = estado_counts.get(INFECTED, 0)
    R = estado_counts.get(RECOVERED, 0)
    
    return S, I, R

def grid_a_rgb(grid):
    """Convierte el grid de estados a una imagen RGB"""
    M, N = grid.shape
    img = np.zeros((M, N, 3))
    
    for estado, color in colors.items():
        mask = (grid == estado)
        img[mask] = color
    
    return img

# Inicializar
grid = inicializar_grid(M, N, I0)
historia_S, historia_I, historia_R = [], [], []

# Configurar la figura
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3)

# Panel izquierdo: Grid espacial
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(f'Propagación Espacial (t = 0)', fontsize=12, fontweight='bold')
ax1.axis('off')
img_display = ax1.imshow(grid_a_rgb(grid), interpolation='nearest')

# Leyenda para el grid
legend_elements = [
    Patch(facecolor=colors[SUSCEPTIBLE], label='Susceptibles'),
    Patch(facecolor=colors[INFECTED], label='Infectados'),
    Patch(facecolor=colors[RECOVERED], label='Recuperados')
]
ax1.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

# Panel derecho: Gráfica temporal
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('Evolución Temporal de Poblaciones', fontsize=12, fontweight='bold')
ax2.set_xlabel('Tiempo (t)', fontsize=10)
ax2.set_ylabel('Población', fontsize=10)
ax2.set_xlim(0, T)
ax2.set_ylim(0, M * N)
ax2.grid(True, alpha=0.3)

line_S, = ax2.plot([], [], 'b-', linewidth=2, label='Susceptibles (S)')
line_I, = ax2.plot([], [], 'r-', linewidth=2, label='Infectados (I)')
line_R, = ax2.plot([], [], 'g-', linewidth=2, label='Recuperados (R)')
ax2.legend(loc='best', frameon=True)

# Añadir información de parámetros
param_text = (f'Parámetros: M×N={M}×{N}, I₀={I0}, β={beta}, γ={gamma}, r={r}')
fig.text(0.5, 0.02, param_text, ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

def animate(frame):
    """Función de animación"""
    global grid
    
    if frame > 0:
        grid = actualizar_grid(grid, beta, gamma, r)
    
    # Contar estados
    S, I, R = contar_estados(grid)
    historia_S.append(S)
    historia_I.append(I)
    historia_R.append(R)
    
    # Actualizar grid espacial
    img_display.set_array(grid_a_rgb(grid))
    ax1.set_title(f'Propagación Espacial (t = {frame})', 
                  fontsize=12, fontweight='bold')
    
    # Actualizar gráfica temporal
    tiempos = list(range(len(historia_S)))
    line_S.set_data(tiempos, historia_S)
    line_I.set_data(tiempos, historia_I)
    line_R.set_data(tiempos, historia_R)
    
    return img_display, line_S, line_I, line_R

# Crear animación
anim = animation.FuncAnimation(fig, animate, frames=T+1, 
                              interval=50, blit=True, repeat=False)

plt.tight_layout()
plt.show()

# Opcional: Guardar la animación
# anim.save('sir_automata.gif', writer='pillow', fps=20)

# Mostrar estadísticas finales
print(f"\n{'='*50}")
print(f"ESTADÍSTICAS FINALES (t = {T})")
print(f"{'='*50}")
print(f"Susceptibles finales: {historia_S[-1]} ({100*historia_S[-1]/(M*N):.1f}%)")
print(f"Infectados finales: {historia_I[-1]} ({100*historia_I[-1]/(M*N):.1f}%)")
print(f"Recuperados finales: {historia_R[-1]} ({100*historia_R[-1]/(M*N):.1f}%)")
print(f"Pico de infectados: {max(historia_I)} en t = {historia_I.index(max(historia_I))}")
print(f"{'='*50}\n")