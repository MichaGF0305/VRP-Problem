# 🚚 **Vehicle Routing Problem (VRP) - Optimización con Colonia de Hormigas (ACO) y Clusterización** 🐜

Este proyecto resuelve el **Problema de Enrutamiento de Vehículos Capacitados (CVRP)** utilizando un enfoque de **optimización por colonia de hormigas (ACO)** combinado con **clusterización KMeans** con restricciones de capacidad.

---

## 📋 **Descripción**

El **Problema de Enrutamiento de Vehículos Capacitados (CVRP)** consiste en encontrar la mejor manera de asignar rutas a una flota de vehículos para servir a un conjunto de clientes, respetando las capacidades de los vehículos y minimizando el costo total del recorrido. Este proyecto utiliza el **Algoritmo de Optimización por Colonia de Hormigas (ACO)** para encontrar soluciones eficientes, y **KMeans** con restricciones para agrupar clientes de acuerdo con la capacidad del vehículo.

### 🧠 **Enfoque Principal:**
1. **Clusterización con KMeansConstrained**: Agrupar los nodos (clientes) en clusters respetando las restricciones de capacidad de los vehículos.
2. **Optimización con ACO**: Aplicar el algoritmo ACO para generar las mejores rutas para cada cluster.
3. **Evaluación y Comparación**: Comparar el rendimiento del algoritmo con métodos tradicionales como la fuerza bruta y el algoritmo Clark-Wright.

---

## 🔧 **Instalación y Requisitos**

### Requisitos:
- Python 3.7+
- Librerías necesarias:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

---

## 🚀 **Uso**

1. **Cargar los Datos**: Carga los datos de los clientes, la matriz de distancias y las capacidades de los vehículos desde archivos en formato Parquet.
   
2. **Aplicar KMeansConstrained**: Instancia y ajusta el modelo de clusterización con la capacidad del vehículo.
   
3. **Ejecutar ACO**: Genera las rutas óptimas para cada cluster utilizando ACO.

4. **Almacenamiento de Resultados**: Los resultados se almacenan en un DataFrame de Pandas con las mejores rutas y costos obtenidos.

### 📝 **Ejemplo de Uso**:

```python
import pandas as pd
from aco import ACO
from kmeans_constrained import KMeansConstrained

# Definir el DataFrame con los problemas a resolver
problem_set_df = pd.read_csv('problems.csv')

benchmark = {}

for idx, row in problem_set_df.iterrows():
    grafo_actual, _, matriz_distancia_actual = get_parquet(index=idx)
    capacidad_actual = row['vehicle_capacity']
    clusterizacion = KMeansConstrained(df=grafo_actual, n_initial_clusters=3, capacity=capacidad_actual)
    clusterizacion.fit_constrained()
    df_nodos = clusterizacion.df_points

    tau_inicial_global = pd.DataFrame(1, index=matriz_distancia_actual.index, columns=matriz_distancia_actual.columns)

    aco_instance = ACO(
        epochs=5,
        k=10,
        tau=tau_inicial_global,
        distance_matrix=matriz_distancia_actual,
        nodos_clusters=df_nodos,
        alpha=1.0,
        beta=2.0,
        Q=100,
        tasa_evap=0.1,
        retornar_al_deposito=True
    )

    mejores_soluciones = aco_instance.run()

    total_costos = []
    rutas_recorridas = []
    for cluster_id, solution in mejores_soluciones.items():
        total_costos.append(solution['mejor_costo'])
        rutas_recorridas.append(solution['mejor_ruta'])

    benchmark[idx] = {
        'problem_cluster': grafo_actual,
        'distance_matrix': row['distance_matrix'],
        'capacidad': capacidad_actual,
        'ACO_best_routes': rutas_recorridas,
        'ACO_best_value': sum(total_costos)
    }

benchmark_df = pd.DataFrame.from_dict(benchmark, orient='index')
```

---

## ⚙️ Estructura del Proyecto

* **src/**: Contiene los scripts principales del algoritmo.

- `aco.py:` Implementación del algoritmo de colonia de hormigas (ACO).

- `kmeans_constrained.py`: Implementación de la clusterización KMeans con restricciones de capacidad.

- `utils.py`: Funciones auxiliares como la carga de datos y almacenamiento de resultados.

* **data/**: Carpeta con los archivos de datos de entrada (en formato Parquet).

- `problems.csv`: Archivo con las configuraciones de problemas a resolver.

* **results/**: Carpeta para guardar los resultados del algoritmo.

---

## 🏆 Métricas de Evaluación

Se han utilizado varias métricas para evaluar el rendimiento del algoritmo:

1. **Error Absoluto (AE):** Valor absoluto de la diferencia entre el valor observado y el valor predicho por el algoritmo.

2. **Error Cuadrado Medio (MSE):** Promedio de los errores al cuadrado, que penaliza errores grandes.

3. **Tiempo de Ejecución:** Comparación del tiempo necesario para encontrar las rutas óptimas.

4. **Uso de Memoria:** Comparación del uso de memoria entre los diferentes algoritmos.

---

## 💡 Conclusiones

El algoritmo propuesto (ACO con clusterización) es significativamente más eficiente en términos de **tiempo** y **memoria** comparado con métodos tradicionales como la **fuerza bruta** y **Clark-Wright**. Sin embargo, el **MSE** alto indica que algunas rutas generadas presentan costos significativamente más altos que las rutas de referencia, lo que puede mejorar ajustando los parámetros del algoritmo o utilizando técnicas adicionales para manejar outliers.

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.