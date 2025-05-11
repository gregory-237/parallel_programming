import matplotlib.pyplot as plt

with open('data/stats.txt', 'r') as file:
    data = {}
    for line in file:
        size, time = line.strip().split(': ')
        data[int(size)] = float(time)

sizes = sorted(data.keys())
times = [data[size] for size in sizes]

plt.figure(figsize=(10, 6))
plt.plot(sizes, times)
plt.xlabel('Размер')
plt.ylabel('Время, мс')
plt.title('Зависимость времени от размера')
plt.grid(True)
plt.savefig('graphic_1.png')