# Lab 06: Simulación del Modelo SIR
## Modelación y Simulación 2025

### Descripción General

Este laboratorio implementa simulaciones computacionales del **modelo epidemiológico SIR** (Susceptible-Infectado-Recuperado) de Kermack-McKendrick usando dos enfoques metodológicos diferentes:

1. **Simulación basada en partículas** - Individuos que se mueven en un espacio 2D
2. **Autómata celular** - Grilla discreta con propagación por vecindad

### Objetivos

- Comparar diferentes representaciones computacionales del mismo modelo epidemiológico
- Analizar el impacto de efectos espaciales en la dinámica de propagación
- Implementar métodos de promediado estadístico para reducir variabilidad estocástica
- Validar simulaciones contra predicciones teóricas del modelo SIR clásico

### Marco Teórico

El modelo SIR se basa en el sistema de ecuaciones diferenciales:

```
dS/dt = -β/N * S * I
dI/dt = β/N * S * I - γ * I  
dR/dt = γ * I
```

Donde:
- **S(t)**: Población susceptible
- **I(t)**: Población infectada  
- **R(t)**: Población recuperada
- **β**: Tasa de transmisión
- **γ**: Tasa de recuperación

### Metodologías Implementadas

#### 1. Modelo de Partículas
- **Dominio**: Cuadrado [0,L] × [0,L] con rebotes en fronteras
- **Movimiento**: Velocidad constante con dirección aleatoria
- **Transmisión**: Por proximidad espacial (radio r)
- **Estados**: 0=Susceptible, 1=Infectado, 2=Recuperado

#### 2. Autómata Celular  
- **Dominio**: Grilla M×N discreta
- **Transmisión**: Por vecindad (radio r)
- **Propagación**: Frentes de onda geográficos
- **Estados**: 0=Susceptible, 1=Infectado, 2=Recuperado

###  Resultados Principales

#### Comparación de Dinámicas

| Aspecto | Partículas | Autómata Celular |
|---------|------------|------------------|
| Población | 100 individuos | 10,000 celdas |
| Tiempo al pico | t ≈ 20 | t ≈ 300 |
| Velocidad | Propagación rápida | Propagación gradual |
| Patrón espacial | Mezcla homogénea | Frentes de onda |
| % final infectado | 98.6% | 97.2% |


### 🛠️ Implementación

#### Características Técnicas
- **Lenguaje**: Python 3.x
- **Bibliotecas**: NumPy, Matplotlib, SciPy (opcional para KDTree)
- **Visualización**: Animaciones interactivas con matplotlib.animation
- **Optimización**: Uso de cKDTree para búsqueda eficiente de vecinos
- **Reproducibilidad**: Semillas aleatorias fijas para comparabilidad

#### Archivos Principales
```
├── ejercicio1.py          # Simulación individual de partículas
├── ejercicio2.py          # Simulación individual de automata celular
├── Ejercicio3_Particulas.py   # Simulación promediada de partículas  
├── Ejercicio3_Particulas.py    # Implementación de autómata celular
└── README.md              # Este archivo
```

### 📈 Análisis Estadístico

- **Repeticiones**: 20 simulaciones por configuración
- **Métricas**: Promedio y desviación estándar de S(t), I(t), R(t)
- **Validación**: Conservación de población total
- **Snapshots**: Análisis en tiempos específicos de evolución


---

**Fecha**: 14 de octubre, 2025  
**Curso**: Modelación y Simulación 2025  
