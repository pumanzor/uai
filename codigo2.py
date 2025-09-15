import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import pandas as pd
from typing import List, Dict, Any

class SolarEnergyOptimizer:
    """
    Clase para optimizar el uso de energía solar con algoritmos genéticos
    Los datos son completamente parametrizables para diferentes escenarios
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el optimizador con configuración parametrizada
        
        Args:
            config: Diccionario con configuración del sistema
                - n_houses: Número de casas a optimizar
                - C_bat: Capacidad de batería (kWh)
                - eta: Eficiencia de carga/descarga
                - D_min: Demanda mínima por hora (kWh)
                - V_max: Límite de inyección a red (kWh)
                - alpha: Peso para autoconsumo
                - beta, gamma, delta: Coeficientes de penalización
                - data_source: Fuente de datos ('synthetic', 'csv', 'api')
                - data_params: Parámetros para carga de datos
        """
        self.config = config
        self.n_houses = config.get('n_houses', 1)
        
        # Parámetros del sistema
        self.C_bat = config.get('C_bat', 10)  # kWh
        self.eta = config.get('eta', 0.9)     # Eficiencia
        self.D_min = config.get('D_min', 0.5)  # kWh
        self.V_max = config.get('V_max', 5.0)  # kWh
        self.alpha = config.get('alpha', 0.1)  # Peso autoconsumo
        self.beta, self.gamma, self.delta = config.get('beta', 100), config.get('gamma', 100), config.get('delta', 100)
        
        # Cargar datos según la fuente especificada
        self.load_data(config.get('data_source', 'synthetic'), config.get('data_params', {}))
        
        # Configurar algoritmo genético
        self.setup_ga()
    
    def load_data(self, data_source: str, data_params: Dict[str, Any]):
        """
        Carga datos de generación, demanda y precios desde diferentes fuentes
        
        Args:
            data_source: 'synthetic', 'csv', 'api' o 'random'
            data_params: Parámetros específicos para la fuente de datos
        """
        if data_source == 'synthetic':
            # Datos sintéticos para demostración (24 horas)
            np.random.seed(42)
            
            # Generación solar - típico día soleado
            self.E_gen = np.array([0, 0, 0, 0, 0.2, 0.8, 2.1, 3.5, 4.8, 5.6, 
                                  6.2, 6.5, 6.3, 5.8, 4.9, 3.8, 2.5, 1.2, 0.5, 0, 0, 0, 0, 0]) * 2
            
            # Demanda energética - patrón típico residencial
            self.E_dem = np.array([0.8, 0.7, 0.6, 0.6, 0.7, 0.9, 1.2, 1.3, 1.1, 1.0, 
                                  1.0, 1.1, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.0, 1.6, 1.3, 1.0, 0.9]) * 1.5
            
            # Precios de compra - variación típica por hora
            self.P_compra = np.array([0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.18, 0.20, 0.19, 0.17, 
                                     0.16, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.22, 0.20, 0.18, 0.16, 0.14])
            
            # Precio de venta - generalmente menor que compra
            self.P_venta = np.full(24, 0.08)  # Precio fijo de venta
            
        elif data_source == 'csv':
            # Cargar desde archivos CSV
            gen_df = pd.read_csv(data_params.get('gen_file', 'generacion_solar.csv'))
            dem_df = pd.read_csv(data_params.get('dem_file', 'demanda.csv'))
            price_df = pd.read_csv(data_params.get('price_file', 'precios.csv'))
            
            self.E_gen = gen_df['generacion'].values
            self.E_dem = dem_df['demanda'].values
            self.P_compra = price_df['precio_compra'].values
            self.P_venta = price_df['precio_venta'].values if 'precio_venta' in price_df.columns else np.full(24, 0.08)
            
        elif data_source == 'api':
            # Aquí se implementaría la conexión a APIs como NASA Power o APIs de precios de energía
            # Por ahora, usamos datos sintéticos como placeholder
            self.load_data('synthetic', {})
            
        elif data_source == 'random':
            # Datos aleatorios para testing
            np.random.seed(data_params.get('seed', 42))
            self.E_gen = np.random.uniform(0, 7, 24)
            self.E_dem = np.random.uniform(0.5, 2.5, 24)
            self.P_compra = np.random.uniform(0.1, 0.25, 24)
            self.P_venta = np.full(24, 0.08)
        
        # Extender datos para múltiples casas si es necesario
        if self.n_houses > 1:
            self.E_gen = np.tile(self.E_gen, (self.n_houses, 1))
            self.E_dem = np.tile(self.E_dem, (self.n_houses, 1))
    
    def setup_ga(self):
        """Configura el algoritmo genético con DEAP"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.eval_fitness)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.01)
    
    def create_individual(self):
        """Crea un individuo con estrategias para 24 horas"""
        ind = []
        for _ in range(24 * self.n_houses):
            u = random.random()
            a = random.random()
            v = random.random()
            total = u + a + v
            ind.extend([u/total, a/total, v/total])
        return creator.Individual(ind)
    
    def eval_fitness(self, individual):
        """Evalúa la aptitud de un individuo (solución)"""
        total_cost = 0
        total_autoconsumo = 0
        total_penalties = 0
        
        # Para cada casa en el sistema
        for house in range(self.n_houses):
            E_bat = self.C_bat / 2  # Inicia con 50% de carga
            cost = 0
            autoconsumo = 0
            penalties = 0
            
            # Para cada hora del día
            for t in range(24):
                # Índice en el cromosoma (depende de si hay múltiples casas)
                idx = (house * 24 + t) * 3
                u, a, v = individual[idx], individual[idx+1], individual[idx+2]
                
                # Obtener datos para esta casa y hora
                if self.n_houses > 1:
                    E_gen = self.E_gen[house, t]
                    E_dem = self.E_dem[house, t]
                else:
                    E_gen = self.E_gen[t]
                    E_dem = self.E_dem[t]
                
                P_compra = self.P_compra[t]
                P_venta = self.P_venta[t]
                
                # Energías asignadas
                U_direct = u * E_gen
                A_almacenada = a * E_gen
                V_venta = v * E_gen
                
                # Balance energético
                demanda_restante = max(0, E_dem - U_direct)
                E_descarga = min(E_bat * self.eta, demanda_restante)
                E_comprada = max(0, demanda_restante - E_descarga)
                
                # Actualizar batería
                E_bat += self.eta * A_almacenada - E_descarga / self.eta
                E_bat = max(0, min(self.C_bat, E_bat))
                
                # Cálculo de penalizaciones
                if E_bat > self.C_bat:
                    penalties += self.beta * (E_bat - self.C_bat)**2
                if (U_direct + E_descarga + E_comprada) < self.D_min:
                    penalties += self.gamma * (self.D_min - (U_direct + E_descarga + E_comprada))**2
                if V_venta > self.V_max:
                    penalties += self.delta * (V_venta - self.V_max)**2
                
                # Costos y autoconsumo
                cost += E_comprada * P_compra - V_venta * P_venta
                autoconsumo += U_direct
            
            total_cost += cost
            total_autoconsumo += autoconsumo
            total_penalties += penalties
        
        fitness = -total_cost + self.alpha * total_autoconsumo - total_penalties
        return fitness,
    
    def run_optimization(self, n_pop=100, n_gen=200, cxpb=0.7, mutpb=0.2):
        """Ejecuta el algoritmo genético"""
        pop = self.toolbox.population(n=n_pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, 
                                      ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
        return pop, log, hof

    def analizar_resultados(self, mejor_ind):
        """Analiza y visualiza los resultados de la optimización"""
        n_houses = self.n_houses
        E_bat_hist = np.zeros((n_houses, 25))
        costos_hora = np.zeros((n_houses, 24))
        autoconsumo_hora = np.zeros((n_houses, 24))
        
        for house in range(n_houses):
            E_bat_hist[house, 0] = self.C_bat / 2  # 50% de carga inicial
            
            for t in range(24):
                # Índice en el cromosoma
                idx = (house * 24 + t) * 3
                u, a, v = mejor_ind[idx], mejor_ind[idx+1], mejor_ind[idx+2]
                
                # Obtener datos para esta casa y hora
                if n_houses > 1:
                    E_gen = self.E_gen[house, t]
                    E_dem = self.E_dem[house, t]
                else:
                    E_gen = self.E_gen[t]
                    E_dem = self.E_dem[t]
                
                P_compra = self.P_compra[t]
                P_venta = self.P_venta[t]
                
                # Energías asignadas
                U_direct = u * E_gen
                A_almacenada = a * E_gen
                V_venta = v * E_gen
                
                # Balance energético
                demanda_restante = max(0, E_dem - U_direct)
                E_descarga = min(E_bat_hist[house, t] * self.eta, demanda_restante)
                E_comprada = max(0, demanda_restante - E_descarga)
                
                # Actualizar batería
                E_bat_hist[house, t+1] = E_bat_hist[house, t] + self.eta * A_almacenada - E_descarga / self.eta
                E_bat_hist[house, t+1] = max(0, min(self.C_bat, E_bat_hist[house, t+1]))
                
                # Calcular costos y autoconsumo
                costos_hora[house, t] = E_comprada * P_compra - V_venta * P_venta
                autoconsumo_hora[house, t] = U_direct
        
        # Visualizaciones para la primera casa (ejemplo)
        if n_houses >= 1:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(3, 2, 1)
            plt.plot(self.E_gen[0] if n_houses > 1 else self.E_gen, 'y-', label='Generación solar')
            plt.plot(self.E_dem[0] if n_houses > 1 else self.E_dem, 'r-', label='Demanda')
            plt.title('Generación y Demanda Energética - Casa 1')
            plt.legend()
            
            plt.subplot(3, 2, 2)
            plt.plot(self.P_compra, 'b-', label='Precio compra')
            plt.plot(self.P_venta, 'g--', label='Precio venta')
            plt.title('Precios de Energía')
            plt.legend()
            
            plt.subplot(3, 2, 3)
            plt.plot(E_bat_hist[0, :-1], 'purple')
            plt.title('Estado de Batería - Casa 1')
            plt.ylim(0, self.C_bat)
            
            plt.subplot(3, 2, 4)
            plt.bar(range(24), costos_hora[0])
            plt.title('Costos por Hora - Casa 1')
            
            plt.subplot(3, 2, 5)
            plt.plot(np.cumsum(costos_hora[0]), 'r-')
            plt.title('Costos Acumulados - Casa 1')
            
            plt.subplot(3, 2, 6)
            horas = range(24)
            u_vals = [mejor_ind[(0*24+t)*3] * (self.E_gen[0,t] if n_houses > 1 else self.E_gen[t]) for t in range(24)]
            a_vals = [mejor_ind[(0*24+t)*3+1] * (self.E_gen[0,t] if n_houses > 1 else self.E_gen[t]) for t in range(24)]
            v_vals = [mejor_ind[(0*24+t)*3+2] * (self.E_gen[0,t] if n_houses > 1 else self.E_gen[t]) for t in range(24)]
            
            plt.stackplot(horas, u_vals, a_vals, v_vals, labels=['Uso directo', 'Almacenamiento', 'Venta'])
            plt.title('Distribución de Energía - Casa 1')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('resultados_optimizacion.png')
            plt.show()
        
        return E_bat_hist, costos_hora, autoconsumo_hora

# Configuración para una casa
config_1_casa = {
    'n_houses': 1,
    'C_bat': 10,
    'eta': 0.9,
    'D_min': 0.5,
    'V_max': 5.0,
    'alpha': 0.1,
    'beta': 100,
    'gamma': 100,
    'delta': 100,
    'data_source': 'synthetic',
    'data_params': {}
}

# Configuración para 100 casas (datos similares pero escalados)
config_100_casas = {
    'n_houses': 100,
    'C_bat': 10,  # Por casa
    'eta': 0.9,
    'D_min': 0.5,
    'V_max': 5.0,
    'alpha': 0.1,
    'beta': 100,
    'gamma': 100,
    'delta': 100,
    'data_source': 'synthetic',
    'data_params': {}
}

# Ejemplo de uso para una casa
print("Iniciando optimización para una casa...")
optimizador = SolarEnergyOptimizer(config_1_casa)
pop, log, hof = optimizador.run_optimization(n_pop=100, n_gen=200)
mejor_ind = hof[0]

# Analizar resultados
print("\nAnalizando resultados...")
E_bat_hist, costos_hora, autoconsumo_hora = optimizador.analizar_resultados(mejor_ind)

# Calcular métricas de rendimiento
costo_total = np.sum(costos_hora)
autoconsumo_total = np.sum(autoconsumo_hora)
costo_referencia = np.sum(optimizador.E_dem * optimizador.P_compra) if optimizador.n_houses == 1 else np.sum(np.sum(optimizador.E_dem, axis=1) * optimizador.P_compra)
ahorro_percentual = (1 - costo_total / costo_referencia) * 100

print(f"\nResultados para {optimizador.n_houses} casa(s):")
print(f"Costo total: €{costo_total:.2f}")
print(f"Costo de referencia (sin optimización): €{costo_referencia:.2f}")
print(f"Autoconsumo total: {autoconsumo_total:.2f} kWh")
print(f"Ahorro estimado: {ahorro_percentual:.1f}%")

# Mostrar evolución del fitness durante las generaciones
if log:
    plt.figure(figsize=(10, 6))
    gen = log.select('gen')
    avg_fitness = log.select('avg')
    max_fitness = log.select('max')
    
    plt.plot(gen, avg_fitness, 'b-', label="Fitness promedio")
    plt.plot(gen, max_fitness, 'r-', label="Fitness máximo")
    plt.title('Evolución del Fitness durante las Generaciones')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig('evolucion_fitness.png')
    plt.show()

# Ejemplo de uso para 100 casas (comentado por tiempo de ejecución)
"""
print("Iniciando optimización para 100 casas...")
optimizador_100 = SolarEnergyOptimizer(config_100_casas)
pop_100, log_100, hof_100 = optimizador_100.run_optimization(n_pop=100, n_gen=50)  # Menos generaciones por rendimiento
mejor_ind_100 = hof_100[0]

# Calcular métricas para 100 casas
E_bat_hist_100, costos_hora_100, autoconsumo_hora_100 = optimizador_100.analizar_resultados(mejor_ind_100)
costo_total_100 = np.sum(costos_hora_100)
autoconsumo_total_100 = np.sum(autoconsumo_hora_100)
costo_referencia_100 = np.sum(np.sum(optimizador_100.E_dem, axis=1) * optimizador_100.P_compra)
ahorro_percentual_100 = (1 - costo_total_100 / costo_referencia_100) * 100

print(f"\nResultados para {optimizador_100.n_houses} casas:")
print(f"Costo total: €{costo_total_100:.2f}")
print(f"Costo de referencia (sin optimización): €{costo_referencia_100:.2f}")
print(f"Autoconsumo total: {autoconsumo_total_100:.2f} kWh")
print(f"Ahorro estimado: {ahorro_percentual_100:.1f}%")
"""
