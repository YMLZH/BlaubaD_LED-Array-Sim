# -------------------------------
# MAIN SCRIPT: LED GRID IRRADIANCE SIMULATION
# EXE-COMPATIBLE - If script is converted to .exe, paths are handled accordingly.
# -------------------------------

# import necessary libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.interpolate import interp1d
from visualization import plot_combined_heatmap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# HANDLE PATHS FOR .EXE OR PYTHON
# -------------------------------
if getattr(sys, 'frozen', False):
    # Running as EXE
    base_path = Path(sys._MEIPASS)
    exe_path = Path(sys.executable).parent
else:
    # Running as normal Python
    base_path = Path(__file__).parent
    exe_path = base_path

input_folder = base_path / "input_data"  # default folder for Excel files

# -------------------------------
# FUNCTION: READ LED DATA FROM EXCEL
# -------------------------------
def read_led_data_from_excel():
    """
    Opens a file dialog to select Excel file and reads LED parameters.
    Returns a list of dicts with x, y, distance, power, and intensities.
    """
    root = tk.Tk()
    root.withdraw()

    while True:
        filepath = filedialog.askopenfilename(
            initialdir=input_folder,
            title="Excel-Datei auswählen",
            filetypes=[("Excel-Dateien", "*.xlsx *.xls")]
        )
        if not filepath:
            messagebox.showerror("Fehler", "Keine Datei ausgewählt.")
        else:
            try:
                df = pd.read_excel(filepath, engine="openpyxl")
                if df.shape[1] < 15:
                    raise ValueError("Die Datei enthält nicht genügend Spalten.")

                x_col_index = 1
                y_col_index = 2
                distance_col_index = 3
                power_col_index = 4
                intensity_start_col_index = 6

                leds = []
                for _, row in df.iterrows():
                    led_info = {
                        "x": float(row.iloc[x_col_index]),
                        "y": float(row.iloc[y_col_index]),
                        "distance": float(row.iloc[distance_col_index]),
                        "power": float(row.iloc[power_col_index]),
                        "intensities": [row.iloc[intensity_start_col_index + i] for i in range(9)],
                    }
                    leds.append(led_info)

                # Return both LED data and Excel filename
                return leds, Path(filepath).stem
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

# -------------------------------
# FUNCTION: CREATE INTENSITY FUNCTION
# -------------------------------
def create_intensity_function(intensities, led_power):
    """
    Creates an I(theta) function [W/sr] from relative intensity values.
    Normalizes distribution to the total LED power.
    Formulas:
    2π ∫[0 to π/2] I(theta) * sin(theta) dtheta = LED Power
    E = I(theta) * cos(theta) / r^2
    """
    angles = np.arange(0, 90, 10)
    relative_intensity = np.array(intensities, dtype=float) / 100.0
    f_rel = interp1d(angles, relative_intensity, kind='linear', bounds_error=False, fill_value=0)

    theta_grid = np.linspace(0, np.pi/2, 500)
    rel_vals = f_rel(np.degrees(theta_grid))
    norm_factor = 2 * np.pi * np.trapz(rel_vals * np.sin(theta_grid), theta_grid)

    if norm_factor == 0:
        raise ValueError("Normierungsfaktor ist 0 – Intensitätsdaten fehlerhaft?")

    scale = led_power / norm_factor

    def intensity_func(theta_deg):
        rel_val = f_rel(theta_deg)
        return scale * rel_val  # W/sr

    return intensity_func

# -------------------------------
# FUNCTION: CALCULATE IRRADIANCE GRID
# -------------------------------
def calc_irradiance_grid(leds, x_coords, y_coords, fine_step=0.1):
    led_x = np.array([l["x"] for l in leds])
    led_y = np.array([l["y"] for l in leds])
    led_z = np.array([l["distance"] for l in leds])

    intensity_funcs = [create_intensity_function(l["intensities"], l["power"]) for l in leds]
    """
    Formula:
        E_gesamt = Σ (I(theta) * cos(theta) / r^2)
    """
    def total_E(points_xy):
        dx = points_xy[:, [0]] - led_x[None, :]
        dy = points_xy[:, [1]] - led_y[None, :]
        r = np.sqrt(dx**2 + dy**2 + led_z[None, :]**2)
        r = np.maximum(r, 1e-9)
        cos_theta = np.clip(led_z[None, :] / r, 0, 1)
        theta_deg = np.degrees(np.arccos(cos_theta))

        E = np.zeros(points_xy.shape[0])
        for i in range(len(leds)):
            I_theta = intensity_funcs[i](theta_deg[:, i])
            E += I_theta * cos_theta[:, i] / (r[:, i]**2)
        return E

    nxg, nyg = len(x_coords), len(y_coords)
    E_mean = np.zeros((nxg, nyg))
    E_min = np.full((nxg, nyg), np.inf)
    E_max = np.zeros((nxg, nyg))

    for i, x in enumerate(x_coords):
        hx = (x_coords[1] - x_coords[0]) / 2.0
        hy = (y_coords[1] - y_coords[0]) / 2.0
        nx = max(1, int(round((x_coords[1] - x_coords[0]) / fine_step)))
        ny = max(1, int(round((y_coords[1] - y_coords[0]) / fine_step)))
        sx, sy = (x_coords[1] - x_coords[0]) / nx, (y_coords[1] - y_coords[0]) / ny
        off_x = (np.arange(nx) + 0.5) * sx - hx
        off_y = (np.arange(ny) + 0.5) * sy - hy

        x_inner = x + off_x
        x_left = np.full(ny, x - hx)
        x_right = np.full(ny, x + hx)
        for j, y in enumerate(y_coords):
            y_inner = y + off_y
            XX, YY = np.meshgrid(x_inner, y_inner, indexing="xy")
            pts = np.column_stack([XX.ravel(), YY.ravel()])
            E_cell = total_E(pts)
            E_mean[i, j] = E_cell.mean()

            Yv = y_inner
            pts_vl = np.column_stack([x_left, Yv])
            pts_vr = np.column_stack([x_right, Yv])
            Xh = x_inner
            y_bot = np.full(nx, y - hy)
            y_top = np.full(nx, y + hy)
            pts_hb = np.column_stack([Xh, y_bot])
            pts_ht = np.column_stack([Xh, y_top])
            E_edges = total_E(np.vstack([pts_vl, pts_vr, pts_hb, pts_ht]))
            E_min[i, j] = E_edges.min()
            E_max[i, j] = E_edges.max()

    return E_mean, E_min, E_max

# -------------------------------
# FUNCTION: CREATE GRID
# -------------------------------
def create_grid(x_min, x_max, y_min, y_max, cell_size=1.0):
    x_coords = np.arange(x_min + 0.5 * cell_size, x_max, cell_size)
    y_coords = np.arange(y_min + 0.5 * cell_size, y_max, cell_size)
    return x_coords, y_coords

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    logging.info("📂 Excel-Datei wird geladen...")
    leds, excel_name = read_led_data_from_excel()
    led_positions = [(led["x"], led["y"]) for led in leds]

    logging.info("🧮 Berechnung wird durchgeführt...")
    x_coords, y_coords = create_grid(-19, 19, -14, 14, cell_size=1.0)
    E_mean, E_min, E_max = calc_irradiance_grid(leds, x_coords, y_coords, fine_step=0.1)

    logging.info("✅ Berechnung abgeschlossen.")

    # -------------------------------
    # CREATE OUTPUT FOLDER BASED ON EXCEL FILE
    # -------------------------------
    output_dir = exe_path / f"output_{excel_name}"
    output_dir.mkdir(exist_ok=True)

    logging.info("📊 Visualisierung wird angezeigt und gespeichert...")
    plot_combined_heatmap(E_mean, E_min, E_max, x_coords, y_coords, led_positions, output_dir)