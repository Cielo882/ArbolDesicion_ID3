# Árbol de decisión con algoritmo ID3

Este repositorio contiene una implementación del algoritmo ID3 (Iterative Dichotomiser 3) , usando entropía como medida de pureza para la construcción de árboles de decisión en Python. 
El árbol de decisión resultante se utiliza para la clasificación de datos.
Para el entrenamiento del modelo se utilizó el conjunto de datos :“CERVICAL CANCER BEHAVIOR RISK” fel repositorio de UCI: https://archive.ics.uci.edu/dataset/537/cervical+cancer+behavior+risk


## Descripción

Para la construcción del modelo se particiono el conjunto de datos como : 
  CONJUNTO DE ENTRENAMIENTO 80% 
  CONJUNTO DE PRUEBA 20%

El código proporcionado consta de varias funciones y una clase que permiten construir un árbol de decisión utilizando el algoritmo ID3. A continuación se describen las principales partes del código:

- `entropia(datos, atributo_objetivo)`: Calcula la entropía de un conjunto de datos para un atributo objetivo dado.
- `ganancia_informacion(datos, atributo, atributo_objetivo)`: Calcula la ganancia de información para un atributo en particular.
- `mejor_atributo(datos, atributos, atributo_objetivo)`: Encuentra el mejor atributo para dividir el conjunto de datos en función de la ganancia de información.
- `construir_arbol(datos, atributos, atributo_objetivo)`: Construye el árbol de decisión utilizando el algoritmo ID3.
- `predecir(arbol, instancia)`: Realiza predicciones utilizando el árbol de decisión construido.
- `evaluar(arbol, datos)`: Evalúa la precisión del modelo utilizando datos de prueba.
- `imprimir_arbol(nodo)`: Función auxiliar para imprimir el árbol de decisión en forma de texto.

## Uso

Para utilizar este código, sigue estos pasos:

1. Asegúrate de tener instaladas las bibliotecas de Python necesarias, como `numpy` y `pandas`.
2. Carga tus datos de entrenamiento y prueba en archivos CSV con el formato adecuado .
3. Puedes usar los datos que vienen en el .zip para esta practica , solo asegurate de ubicarlos en la misma carpeta del Script
4. Especifica los atributos y el atributo objetivo en tu conjunto de datos.
5. Construye el árbol de decisión utilizando la función `construir_arbol`.
6. Evalúa el modelo utilizando la función `evaluar`.
7. Para documentación detallada revisa el reporte en .pdf de esta practica anexado en el repositorio.

## Ejemplo

```python
# Carga los datos de entrenamiento desde "entrenamiento.csv"
datos_entrenamiento = pd.read_csv("entrenamiento.csv")

# Carga los datos de prueba desde "prueba.csv"
datos_prueba = pd.read_csv("prueba.csv")

# Especifica los atributos y el atributo objetivo
atributos = list(datos_entrenamiento.columns[:-1])
atributo_objetivo = datos_entrenamiento.columns[-1]

# Construye el árbol de decisión con los datos de entrenamiento
arbol_decision = construir_arbol(datos_entrenamiento, atributos, atributo_objetivo)

# Evalúa el modelo con los datos de prueba
precision = evaluar(arbol_decision, datos_prueba)
print(f"Precision del modelo: {precision * 100:.2f}%") 
```
## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
