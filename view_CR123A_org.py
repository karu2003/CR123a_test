import csv
import matplotlib.pyplot as plt

# Path to the CSV file
filename = "CR123A.csv"

# Separate lists for each curve
t_300, v_300 = [], []
t_100, v_100 = [], []
t_500, v_500 = [], []

# Reading and filtering
with open(filename, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            t = float(row[0].replace(',', '.'))
            v300 = float(row[1].replace(',', '.'))
            v100 = float(row[2].replace(',', '.'))
            v500 = float(row[3].replace(',', '.'))

            # Add to each graph independently if the value is valid
            if v300 >= 0:
                t_300.append(t)
                v_300.append(v300)
            if v100 >= 0:
                t_100.append(t)
                v_100.append(v100)
            if v500 >= 0:
                t_500.append(t)
                v_500.append(v500)

        except (ValueError, IndexError):
            continue

# Organize data into a dictionary
data_dict = {
    "300 mA": {"time": t_300, "voltage": v_300},
    "100 mA": {"time": t_100, "voltage": v_100},
    "500 mA": {"time": t_500, "voltage": v_500},
}

# Plotting the graph using the dictionary
plt.figure(figsize=(10, 6))
ax = plt.gca()

for label, data in data_dict.items():
    line, = ax.plot(data["time"], data["voltage"], label=label)

    # Annotate the graph with a separate legend
    x_text = data["time"][-1]  # Last time point
    y_text = data["voltage"][-1]  # Last voltage point

    # Add annotation with an arrow pointing to the graph
    ax.annotate(
        label,
        xy=(x_text, y_text),
        xytext=(x_text + 0.5, y_text + 0.1),  # Offset for the text
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=line.get_color(), lw=1.5),
        fontsize=10,
        color=line.get_color(),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line.get_color(), lw=1),
    )

plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
plt.title('CR123A Battery Discharge')
plt.grid(True)
plt.tight_layout()
plt.show()
