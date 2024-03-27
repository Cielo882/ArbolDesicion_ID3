# Version 1.0
import numpy as np
import pandas as pd

# Define la funcion para calcular la entropia
def entropia(datos, atributo_objetivo):
    entropia = 0
    valores = datos[atributo_objetivo].unique()
    for valor in valores:
        p = datos[atributo_objetivo].value_counts()[valor] / len(datos)
        entropia += -p * np.log2(p)
    return round(entropia, 3)  # Redondea la entropia a 3 decimales

# Define la funcion para calcular la ganancia de informacion
def ganancia_informacion(datos, atributo, atributo_objetivo):
    entropia_s = entropia(datos, atributo_objetivo)
    valores = datos[atributo].unique()
    entropia_ponderada = 0
    print(f"Ganancia de Informacion para el atributo '{atributo}':")
    for valor in valores:
        subconjunto = datos[datos[atributo] == valor]
        entropia_subconjunto = entropia(subconjunto, atributo_objetivo)
        entropia_ponderada += (len(subconjunto) / len(datos)) * entropia_subconjunto
        print(f"  Valor '{valor}': Entropia = {round(entropia_subconjunto, 3)}")
    ganancia = entropia_s - entropia_ponderada
    print (entropia_s)
    print(f"Ganancia total para el atributo '{atributo}' = {round(ganancia, 3)}")
    return ganancia

# Define la funcion para encontrar el mejor atributo
def mejor_atributo(datos, atributos, atributo_objetivo):
    ganancias = [ganancia_informacion(datos, atributo, atributo_objetivo) for atributo in atributos]
    mejor_atrib = atributos[np.argmax(ganancias)]
    print("------------------------")
    print("------------------------")
    print("------------------------")


    print(f"Mejor atributo seleccionado: {mejor_atrib}")

    return mejor_atrib

# Define la clase para representar un nodo del arbol
class Nodo:
    def __init__(self, datos, padre, valor=None):
        self.datos = datos
        self.padre = padre
        self.hijos = {}
        self.valor = valor

# Define la funcion para construir el arbol ID3
def construir_arbol(datos, atributos, atributo_objetivo, padre=None, valor=None):
    if len(datos[atributo_objetivo].unique()) == 1:
        return Nodo(datos, padre, datos[atributo_objetivo].iloc[0])

    if not atributos:
        return Nodo(datos, padre, datos[atributo_objetivo].mode().iloc[0])

    mejor_atrib = mejor_atributo(datos, atributos, atributo_objetivo)
    arbol = Nodo(datos, padre, mejor_atrib)  # Usa el mejor atributo como nombre del nodo
    valores = datos[mejor_atrib].unique()
    for val in valores:
        subconjunto = datos[datos[mejor_atrib] == val]
        if len(subconjunto) == 0:
            arbol.hijos[val] = Nodo(datos, arbol, datos[atributo_objetivo].mode().iloc[0])
        else:
            arbol.hijos[val] = construir_arbol(subconjunto, [atributo for atributo in atributos if atributo != mejor_atrib], atributo_objetivo, arbol, val)
    return arbol

# Define la funcion para hacer predicciones con el arbol
def predecir(arbol, instancia):
    if len(arbol.hijos) == 0:
        return arbol.valor
    try:
        return predecir(arbol.hijos[instancia[arbol.datos.columns[-1]]], instancia)
    except KeyError:
        return arbol.datos[arbol.datos.columns[-1]].mode().iloc[0]

# Carga los datos de entrenamiento desde "entrenamiento.csv"
datos_entrenamiento = pd.read_csv("entrenamiento.csv")

# Carga los datos de prueba desde "prueba.csv"
datos_prueba = pd.read_csv("prueba.csv")

# Especifica los atributos y el atributo objetivo
atributos = list(datos_entrenamiento.columns[:-1])
atributo_objetivo = datos_entrenamiento.columns[-1]

# Construye el arbol de decision con los datos de entrenamiento
arbol_decision = construir_arbol(datos_entrenamiento, atributos, atributo_objetivo)

# Funcion para evaluar la precision del modelo
def evaluar(arbol, datos):
    correctos = 0
    total = len(datos)
    for _, fila in datos.iterrows():
        prediccion = predecir(arbol, fila)
        etiqueta_real = fila[atributo_objetivo]
        if prediccion == fila[atributo_objetivo]:
            correctos += 1
        print(f"Etiqueta Real: {etiqueta_real}, Etiqueta Predicha: {prediccion}")

    precision = correctos / total
    return precision

# Evalua el modelo con los datos de prueba
precision = evaluar(arbol_decision, datos_prueba)
print(f"Precision del modelo: {precision * 100:.2f}%")

# Funcion imprimir_arbol para mostrar los nombres de los nodos
def imprimir_arbol(nodo, sangria=""):
    if len(nodo.hijos) == 0:
        print(f"{sangria}Clase: {nodo.valor}")
    else:
        print(f"{sangria}{nodo.valor}")
        for valor, hijo in nodo.hijos.items():
            print(f"{sangria}  {valor} -> ", end="")
            imprimir_arbol(hijo, sangria + "    ")

imprimir_arbol(arbol_decision)

# Retorna el modelo entrenado (el arbol)
modelo_entrenado = arbol_decision
