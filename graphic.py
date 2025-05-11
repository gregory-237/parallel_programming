import matplotlib.pyplot as plt


def read_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            size, time = line.strip().split(': ')
            data[int(size)] = float(time)
    return data


data1 = read_data('data/stats_1.txt')
data2 = read_data('data/stats_2.txt')

sizes1 = sorted(data1.keys())
times1 = [data1[size] for size in sizes1]
sizes2 = sorted(data2.keys())
times2 = [data2[size] for size in sizes2]

plt.figure(figsize=(10, 6))
plt.plot(sizes1, times1, label='Время', marker='o')
plt.plot(sizes2, times2, label='Время с openmp', marker='x')

plt.xlabel('Размер')
plt.ylabel('Время, мс')
plt.title('Зависимость времени от размера')
plt.grid(True)
plt.legend()
plt.savefig('graphic_comparison.png')
plt.show()
