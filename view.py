import os
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.patches as mpatches
import csv
from matplotlib.ticker import FuncFormatter


def read_data_from_file(file_path):
    """Reads data from a text file."""
    with open(file_path, "r") as file:
        data = file.readlines()
    return [float(line.strip()) for line in data]


def calculate_energy_from_interpolation(time, voltage, current):
    """
    Calculates the discharge energy of the battery using interpolated data.

    Parameters:
        time (list): Time points in hours.
        voltage (list): Voltage values corresponding to the time points.
        current (float): Discharge current in amperes.

    Returns:
        float: Total discharge energy in joules.
    """
    time_seconds = [t * 3600 for t in time]  # Convert time from hours to seconds
    voltage_integral = simps(voltage, time_seconds)
    return current * voltage_integral  # Joules

def calculate_energy(voltage_list, time_list, current_a):
    energy_wh = 0.0
    for i in range(len(voltage_list) - 1):
        u = voltage_list[i]
        delta_t = time_list[i+1] - time_list[i]
        energy_wh += u * current_a * delta_t  # Вт·ч
    return energy_wh


def estimate_annotation_size(text, fontsize=9):
    """
    Estimates the approximate size of an annotation based on the length of the text and font size.
    
    Parameters:
        text: Annotation text
        fontsize: Font size
        
    Returns:
        tuple: (width, height) approximate dimensions
    """
    lines = text.split('\n')
    max_line_length = max(len(line) for line in lines)
    
    # Improved coefficients for better size estimation
    width_per_char = 0.015 * (fontsize / 9)  # Increased width per character
    height_per_line = 0.06 * (fontsize / 9)  # Increased height per line
    
    # Add padding
    width = max_line_length * width_per_char + 0.05
    height = len(lines) * height_per_line + 0.02
    
    return width, height

def rectangles_overlap(rect1, rect2, margin=0.08):
    """
    Checks if two rectangles overlap.
    
    Parameters:
        rect1: (x, y, width, height) of the first rectangle
        rect2: (x, y, width, height) of the second rectangle
        margin: additional margin
        
    Returns:
        bool: True if the rectangles overlap
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Add margin to dimensions
    w1 += margin
    h1 += margin
    w2 += margin
    h2 += margin
    
    # Check for overlap
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def find_free_position(text, orig_xy, annotated_rects, fontsize=9):
    """
    Finds a free position for an annotation.
    
    Parameters:
        text: Annotation text
        orig_xy: Original coordinates (x, y)
        annotated_rects: List of rectangles of existing annotations
        fontsize: Font size
        
    Returns:
        tuple: ((x, y), rect) coordinates for placing the annotation and its rectangle
    """
    x_text, y_text = orig_xy
    orig_x, orig_y = orig_xy
    
    # Parameters for search - prioritize upward movement
    vertical_offset = 0.12  # Increased vertical offset
    horizontal_offset = 0.04  # Increased horizontal offset
    max_attempts = 200  # More attempts
    
    attempts = 0
    direction = 1
    
    # Estimate annotation size
    width, height = estimate_annotation_size(text, fontsize)
    current_rect = (x_text, y_text, width, height)
    
    # Store the best position found with minimum overlaps
    best_position = (x_text, y_text)
    best_rect = current_rect
    min_overlaps = float('inf')
    
    # Check for overlap with existing annotations
    while any(rectangles_overlap(current_rect, rect) for rect in annotated_rects) and attempts < max_attempts:
        # First try moving up to avoid overlaps
        if attempts < 30:
            # Prioritize upward movement for first attempts
            y_text += vertical_offset
        else:
            # After initial attempts, use a more varied approach
            if attempts % 3 == 0:
                # Move up
                y_text += vertical_offset
            elif attempts % 3 == 1:
                # Move horizontally
                x_text += direction * horizontal_offset
                direction *= -1
            else:
                # Try diagonal movement
                y_text += vertical_offset * 0.7
                x_text += direction * horizontal_offset * 0.7
        
        # Increase offsets after many attempts
        if attempts % 10 == 0 and attempts > 0:
            vertical_offset *= 1.2
            horizontal_offset *= 1.2
        
        attempts += 1
        
        # Update rectangle
        current_rect = (x_text, y_text, width, height)
        
        # Count overlaps for this position
        overlap_count = sum(1 for rect in annotated_rects if rectangles_overlap(current_rect, rect))
        
        # Keep track of position with minimum overlaps
        if overlap_count < min_overlaps:
            min_overlaps = overlap_count
            best_position = (x_text, y_text)
            best_rect = current_rect
        
        # If no overlaps, we found a good position
        if overlap_count == 0:
            break
    
    # If we couldn't find a perfect position, use the best one
    if attempts >= max_attempts:
        return best_position, best_rect
    
    return (x_text, y_text), current_rect

def find_nearest_point(x_array, y_array, x_target, y_target):
    """
    Finds the nearest point on the graph to the specified target coordinates.
    
    Parameters:
        x_array: Array of x coordinates
        y_array: Array of y coordinates
        x_target: Target x coordinate
        y_target: Target y coordinate
        
    Returns:
        tuple: (x, y) of the nearest point on the graph
    """
    # Calculate distance to each point
    distances = np.sqrt((np.array(x_array) - x_target)**2 + (np.array(y_array) - y_target)**2)
    
    # Find index of the minimum distance
    idx = np.argmin(distances)
    
    # Return the nearest point
    return x_array[idx], y_array[idx]

def estimate_tick_label_width(value, fontsize=10):
    """
    Estimates the width of a tick label based on the value and font size.
    
    Parameters:
        value: The value to be displayed as a tick
        fontsize: Font size of the tick label
        
    Returns:
        float: Estimated width in data units
    """
    # Convert the value to a string with 2 decimal places
    text = f"{value:.2f}"
    
    # Estimate width based on text length and font size
    # These coefficients need to be adjusted based on the actual display
    width_per_char = 0.01 * (fontsize / 10)
    
    # Get text length and calculate width
    text_len = len(text)
    width = text_len * width_per_char
    
    # Return estimated width plus some padding
    return width + 0.01

def filter_overlapping_ticks(ticks, min_distance_factor=1.5, fontsize=10):
    """
    Filters out ticks that would result in overlapping labels.
    Uses a more aggressive approach to reduce the number of ticks.
    
    Parameters:
        ticks: List of tick positions
        min_distance_factor: Factor to multiply the estimated label width by
        fontsize: Font size of tick labels
        
    Returns:
        list: Filtered list of ticks
    """
    if not ticks:
        return []
    
    # Sort ticks
    ticks = sorted(ticks)
    
    # For logarithmic scale, we need to consider relative distances
    # Extract the min and max values to normalize distances
    min_val = min(ticks)
    max_val = max(ticks)
    range_val = max_val - min_val
    
    # Initialize filtered list with the first tick
    filtered_ticks = [ticks[0]]
    
    # Keep track of the "reserved space" for each tick
    # For logarithmic scale, we use relative positions
    label_width = estimate_tick_label_width(ticks[0], fontsize)
    reserved_spaces = [(ticks[0] - label_width/2, ticks[0] + label_width/2)]
    
    # Process remaining ticks
    for tick in ticks[1:]:
        # Calculate the label width for this tick
        label_width = estimate_tick_label_width(tick, fontsize)
        
        # Calculate the space this tick would occupy
        tick_space = (tick - label_width/2, tick + label_width/2)
        
        # Adjust the minimum distance based on how close we are to other ticks
        # More aggressive filtering when ticks are close together
        adjusted_min_distance = min_distance_factor * (1.0 + 0.5 * (tick_space[1] - tick_space[0]) / range_val)
        
        # Check if this space overlaps with any reserved space
        overlaps = False
        for space in reserved_spaces:
            # More strict overlap check
            if not ((space[1] * adjusted_min_distance < tick_space[0]) or 
                    (space[0] > tick_space[1] * adjusted_min_distance)):
                overlaps = True
                break
        
        # If no overlap, add this tick and its space
        if not overlaps:
            filtered_ticks.append(tick)
            reserved_spaces.append(tick_space)
    
    return filtered_ticks

def plot_all_data(files_data, indices, energies, discharge_currents, org_data_dict):
    """
    Plots data from multiple files on the same graph and annotates the energy values.

    Parameters:
        files_data (dict): Dictionary containing data for each file.
        indices (dict): Dictionary containing indices for annotations.
        energies (dict): Dictionary containing energy values in Wh for each file.
        discharge_currents (dict): Dictionary containing discharge currents for each file.
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    log_ticks = []

    # List to store annotation rectangles
    annotated_rects = []

    # Process data from txt files
    for file_name, data in files_data.items():
        time = [i * 0.25 / 3600 for i in range(len(data))]  # Time in hours
        log_ticks.append(time[-1])

        # Interpolation
        interp_func = interp1d(time, data, kind="cubic", fill_value="extrapolate")
        interpolated_time = np.linspace(min(time), max(time), 500)
        interpolated_data = interp_func(interpolated_time)

        # Energy
        discharge_current = extract_current_from_filename(file_name)
        energy_wh_interpolated = calculate_energy_from_interpolation(
            interpolated_time, interpolated_data, discharge_current
        )

        energy_wh = energies[file_name]

        # Plotting the original data and graph
        ax.plot(time, data, ".", label=f"{file_name} (Original Data)", alpha=0.7)
        (line,) = ax.plot(interpolated_time, interpolated_data, "-", label=file_name)

        # Initial annotation position at the end of the line
        x_text = interpolated_time[-1]
        y_text = interpolated_data[-1]
        
        # Prepare annotation text
        label = f"{discharge_current} A\n{energy_wh:.3f} Wh"
        
        # Find free position for annotation
        (x_text, y_text), rect = find_free_position(
            label, (x_text, y_text), annotated_rects, fontsize=9
        )
        
        # Find nearest point on the graph to the annotation position
        nearest_x, nearest_y = find_nearest_point(
            interpolated_time, interpolated_data, x_text, y_text
        )
        
        # Add rectangle to list
        annotated_rects.append(rect)
        
        ax.annotate(
            label,
            xy=(nearest_x, nearest_y),  # Point the arrow to the nearest point
            xytext=(x_text, y_text),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=line.get_color(), lw=1.5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=line.get_color(), lw=1),
            color=line.get_color(),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    # Process data from org_data_dict - use the same algorithm
    for label, data in org_data_dict.items():
        current_a = float(label.split()[0]) / 1000   
        energy = calculate_energy(data["voltage"], data["time"], current_a)
        text_label = f"{label}\n{energy:.3f} Wh"
        (line,) = ax.plot(data["time"], data["voltage"], label=label)
        log_ticks.append(data["time"][-1])

        # Initial annotation position at the end of the line
        x_text = data["time"][-1]
        y_text = data["voltage"][-1]

        # Find free position for annotation
        (x_text, y_text), rect = find_free_position(
            text_label, (x_text, y_text), annotated_rects, fontsize=10
        )
        
        # Find nearest point on the graph to the annotation position
        nearest_x, nearest_y = find_nearest_point(
            data["time"], data["voltage"], x_text, y_text
        )
        
        # Add rectangle to list
        annotated_rects.append(rect)

        # Add annotation 
        ax.annotate(
            text_label,
            xy=(nearest_x, nearest_y),  # Point the arrow to the nearest point
            xytext=(x_text, y_text),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=line.get_color(), lw=1.5),
            fontsize=10,
            color=line.get_color(),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line.get_color(), lw=1),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    ax.set_xscale("log")
 
    # Set custom ticks for the X-axis
    log_ticks.sort()
    # More aggressive filtering of overlapping ticks
    log_ticks = filter_overlapping_ticks(log_ticks, min_distance_factor=1.1)
    ax.set_xticks(log_ticks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Set X-axis limits to match the data range
    # min_time = min(min(data["time"]) for data in org_data_dict.values())
    min_time = 0.04  # Ensure left limit is >= 1e-3
    max_time = max(max(data["time"]) for data in org_data_dict.values())
    ax.set_xlim(
        left=max(min_time, 1e-3), right=max_time
    )  # Ensure left limit is >= 1e-3

    plt.title("Battery CR123A Discharge Graph")
    plt.xlabel("Time (h)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("battery_discharge_graph.png", dpi=300)
    plt.show()


def find_last_index_less_than(arr, x):
    """
    Finds the last index in the array where the value is less than x.

    Parameters:
        arr (list or np.array): Array of values.
        x (float): Threshold value.

    Returns:
        int: Last index where the value is less than x, or -1 if not found.
    """
    arr = np.array(arr)
    indices = np.where(arr < x)[0]  # Find indices where the value is < x
    if len(indices) == 0:
        return -1  # If no such elements exist
    return indices[-1]  # Return the last index


def find_last_index(arr, x):
    """
    Finds the last index in the array where the value equals x.

    Parameters:
        arr (list or np.array): Array of values.
        x (float): Target value.

    Returns:
        int: Last index where the value equals x, or -1 if not found.
    """
    arr = np.array(arr)  # Convert input to a numpy array if it's a list
    indices = np.where(arr == x)[0]  # Find all indices where the value equals x
    if len(indices) == 0:
        return -1  # If x is not found, return -1
    return indices[-1]  # Return the last index


def get_rounding_step(x):
    """
    Determines the rounding step based on the number of decimal places.

    Parameters:
        x (float): Input value.

    Returns:
        float: Rounding step.
    """
    x_str = str(x)
    if "." in x_str:
        decimals = len(x_str.split(".")[-1])
        return 10**-decimals
    return 1  # If x is an integer


def find_last_index_less(arr, x):
    """
    Finds the last index in the array where the quantized value is less than x.

    Parameters:
        arr (list or np.array): Array of values.
        x (float): Threshold value.

    Returns:
        int: Last index where the quantized value is less than x, or -1 if not found.
    """
    arr = np.array(arr, dtype=float)
    step = get_rounding_step(x)

    # Quantize the array
    quantized_arr = np.floor(arr / step) * step
    # Find indices where values are less than x
    indices = np.where(quantized_arr < x)[0]
    # Compute the difference between consecutive indices
    diff = np.diff(indices)
    # Find the last index in diff where the value is not equal to 1
    last_diff_index = -1  # Default to -1
    for i in range(len(diff) - 1, -1, -1):
        if diff[i] != 1:
            last_diff_index = i
            break

    # If found, add it to the first index from indices
    if len(indices) > 0:
        if last_diff_index != -1:
            adjusted_index = indices[last_diff_index + 1]
        else:
            adjusted_index = indices[0]  # Use the first element of indices
        return adjusted_index

    # If nothing is found, return -1
    return -1


def calculate_discharge_energy(data, discharge_current, time_step=0.25):
    """
    Calculates the discharge energy of the battery.

    Parameters:
        data (list): Voltage values over time.
        discharge_current (float): Discharge current in amperes.
        time_step (float): Time step between measurements in seconds.

    Returns:
        float: Total discharge energy in joules.
    """
    energy = 0.0
    for voltage in data:
        energy += voltage * float(discharge_current) * time_step
    return energy


def calculate_discharge_energy_integral(data, discharge_current, time_step=0.25):
    """
    Calculates the discharge energy of the battery using numerical integration.

    Parameters:
        data (list): Voltage values over time.
        discharge_current (float): Discharge current in amperes.
        time_step (float): Time step between measurements in seconds.

    Returns:
        float: Total discharge energy in joules.
    """
    # Time array
    time = np.arange(0, len(data) * time_step, time_step)
    # Integrate voltage * current
    energy = simps([v * float(discharge_current) for v in data], time)
    return energy


def calculate_energy_from_interpolation(
    interpolated_time, interpolated_data, discharge_current
):
    """
    Calculates the discharge energy of the battery using interpolated data.

    Parameters:
        interpolated_time (list): Interpolated time points.
        interpolated_data (list): Interpolated voltage values.
        discharge_current (float): Discharge current in amperes.

    Returns:
        float: Total discharge energy in joules.
    """
    # Integrate voltage * current over time
    energy = simps(
        [v * float(discharge_current) for v in interpolated_data], interpolated_time
    )
    return energy


def extract_current_from_filename(file_name):
    """
    Extracts current value from filename.
    Examples:
    - 1A.txt returns 1.0
    - 1_5A.txt returns 1.5
    - 0_5A.txt returns 0.5
    
    Parameters:
        file_name (str): Name of the file
        
    Returns:
        float: Current value in amperes
    """
    # Remove file extension
    name_without_ext = file_name.split('.')[0]
    # Remove 'A' suffix
    name_without_a = name_without_ext.replace('A', '')
    # Replace underscore with decimal point
    current_str = name_without_a.replace('_', '.')
    # Convert to float
    try:
        return float(current_str)
    except ValueError:
        print(f"Could not extract current from filename: {file_name}")
        return 0.0


if __name__ == "__main__":
    current_directory = os.getcwd()  # Current directory
    txt_files = [f for f in os.listdir(current_directory) if f.endswith(".txt")]
    x_max = 3.3  # Value to find and mark on the graph
    x_min = 0.9  # Value to find and mark on the graph

    # Path to the CSV file
    filename = "CR123A.csv"

    # Separate lists for each curve
    t_300, v_300 = [], []
    t_100, v_100 = [], []
    t_500, v_500 = [], []

    # Reading and filtering
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                t = float(row[0].replace(",", "."))
                v300 = float(row[1].replace(",", "."))
                v100 = float(row[2].replace(",", "."))
                v500 = float(row[3].replace(",", "."))

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

    org_data_dict = {
        "300 mA": {"time": t_300, "voltage": v_300},
        "100 mA": {"time": t_100, "voltage": v_100},
        "500 mA": {"time": t_500, "voltage": v_500},
    }

    if not txt_files:
        print("No .txt files found in the current directory.")
    else:
        files_data = {}
        indices = {}
        energies = {}
        discharge_currents = {}

        for file_name in txt_files:
            try:
                print(f"Reading data from file: {file_name}")
                data = read_data_from_file(file_name)
                discharge_current = extract_current_from_filename(file_name)
                discharge_currents[file_name] = discharge_current
                print(f"Discharge current: {discharge_current}")

                # Find indices for x_max and x_min
                last_index_max = find_last_index(data, x_max)
                last_index_min = find_last_index_less(data, x_min)

                # Store indices with labels
                indices[file_name] = []
                if last_index_max != -1:
                    indices[file_name].append((last_index_max, "x_max"))
                if last_index_min != -1:
                    indices[file_name].append((last_index_min, "x_min"))
                valid_data = data[last_index_max:last_index_min]
                files_data[file_name] = valid_data

                # Calculate discharge energy in joules
                energy_joules = calculate_discharge_energy(
                    valid_data, discharge_current
                )

                # Calculate discharge energy using integration
                energy_joules = calculate_discharge_energy_integral(
                    valid_data, discharge_current
                )
                # Convert energy to watt-hours
                energy_wh = energy_joules / 3600
                energies[file_name] = energy_wh
                print(
                    f"Discharge energy for {file_name}: {energy_joules:.2f} J ({energy_wh:.4f} Wh)"
                )

            except FileNotFoundError:
                print(f"File {file_name} not found.")
            except ValueError:
                print(f"Error: File {file_name} contains invalid data.")

        if files_data:
            plot_all_data(
                files_data, indices, energies, discharge_currents, org_data_dict
            )
