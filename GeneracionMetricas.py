import psutil
import matplotlib.pyplot as plt

# Creamos las listas para almacenar los datos de tiempo y uso de la memoria RAM
time_list = []
mem_list = []

# Inicializamos el gráfico
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Uso de la memoria RAM (%)')

# Definimos la función que actualizará el gráfico
def update_graph():
    # Obtenemos el tiempo actual y el porcentaje de uso de la memoria RAM
    current_time = psutil.cpu_times().system
    current_mem = psutil.virtual_memory().percent
    
    # Agregamos los datos a las listas
    time_list.append(current_time)
    mem_list.append(current_mem)
    
    # Limitamos las listas a 100 puntos para evitar sobrecargar la memoria
    if len(time_list) > 100:
        time_list.pop(0)
        mem_list.pop(0)
    
    # Actualizamos el gráfico
    ax.clear()
    ax.plot(time_list, mem_list)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Uso de la memoria RAM (%)')
    plt.draw()
    
    # Esperamos 1 segundo antes de actualizar de nuevo el gráfico
    fig.canvas.flush_events()
    plt.pause(1)

# Actualizamos el gráfico en un ciclo infinito
while True:
    update_graph()
