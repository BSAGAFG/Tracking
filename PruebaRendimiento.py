import psutil
import matplotlib.pyplot as plt
import os
import GPUtil

# Medición del uso de la CPU
def get_cpu_usage():
    cpu_perc = psutil.cpu_percent()
    return cpu_perc

print(get_cpu_usage())

# Medición del uso de la GPU
def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    gpu_load = []
    for gpu in gpus:
        gpu_load.append(gpu.load*100)

print(get_gpu_usage())


def get_memory_usage():
    mem = psutil.virtual_memory()
    mem_used_perc = mem.used * 100 / mem.total
    return mem

print(get_memory_usage())

def get_network_usage():
    net_io_counters = psutil.net_io_counters()
    return net_io_counters.bytes_sent, net_io_counters.bytes_recv

print(get_network_usage())

def get_disk_usage():
    # partitions = psutil.disk_partitions()
    # used_space = 0
    # total_space = 0
    # for partition in partitions:
    #     try:
    #         usage = psutil.disk_usage(partition.mountpoint)
    #         used_space += usage.used
    #         total_space += usage.total
    #     except:
    #         pass
    # disk_usage = used_space * 100 / total_space
    disk_usage = psutil.disk_usage("/")
    return disk_usage

print(get_disk_usage())

def plot_cpu_memory():
    cpu_perc = get_cpu_usage()
    mem_perc = get_memory_usage()
    fig, ax = plt.subplots()
    ax.bar(['CPU', 'Memory'], [cpu_perc, mem_perc])
    ax.set_ylabel('Usage (%)')
    ax.set_title('CPU and Memory usage')
    plt.show()


# Salida de información
print("--------------CPU--------------")
print(f"Porcentaje de uso de CPU: {get_cpu_usage()}%")
print("--------------GPU--------------")
print("Uso de GPU:", get_gpu_usage(), "%")
print("--------------Memoria RAM--------------")
print(f"Memoria total: {get_memory_usage().total/1000000000}", " Gbytes")
print(f"Memoria disponible: {get_memory_usage().available/1000000000}", " Gbytes")
print(f"Memoria en uso: {get_memory_usage().used/1000000000}", " Gbytes")
print(f"Porcentaje de memoria en uso: {get_memory_usage().percent}%")
print(f"Uso de memoria RAM: {get_memory_usage().percent}% de {get_memory_usage().total / (1024 * 1024)} MB")
print("--------------Uso de la red--------------")
print("Uso de la red - Enviados:", get_network_usage()[0]/1000000, "Mbytes - Recibidos:", get_network_usage()[1]/1000000, "Mbytes")
print("--------------Uso del disco duro--------------")
print("Uso del disco duro - Utilizados:", get_disk_usage().used/1000000, "Mbytes - Total:", get_disk_usage().total/1000000, "Mbytes - Porcentaje:", get_disk_usage().percent, "%")

#plot_cpu_memory()