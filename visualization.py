# -------------------------------
# VISUALIZATION SCRIPT: LED GRID IRRADIANCE SIMULATION
# -------------------------------

# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons
from pathlib import Path
from datetime import datetime

def plot_combined_heatmap(E_mean, E_min, E_max, x_coords, y_coords, led_positions, output_dir: Path):
    """
    Displays interactive heatmap of LED irradiance and saves PNG images.

    Parameters:
        E_mean, E_min, E_max (2D arrays): Irradiance grids
        x_coords, y_coords (1D arrays): Grid coordinates
        led_positions (list of tuples): LED (x,y) positions for markers
        output_dir (Path): Directory to save PNG images
    """

    cell_size_x = x_coords[1] - x_coords[0]
    cell_size_y = y_coords[1] - y_coords[0]
    x_edges = np.append(x_coords, x_coords[-1] + cell_size_x) - cell_size_x / 2
    y_edges = np.append(y_coords, y_coords[-1] + cell_size_y) - cell_size_y / 2

    data_w = x_edges[-1] - x_edges[0]
    data_h = y_edges[-1] - y_edges[0]
    fig_aspect = data_w / data_h
    fig_height = 9
    fig_width = fig_height * fig_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    try:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.state('zoomed')
    except Exception:
        pass

    current_data = {'data': E_mean}
    heatmap = ax.pcolormesh(x_edges, y_edges, current_data['data'].T, shading='auto', cmap='inferno')
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Bestrahlungsstärke (mW/cm²)')

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title('Bestrahlungsstärke Heatmap', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(x_coords[0] - cell_size_x/2, x_coords[-1] + cell_size_x/2)
    ax.set_ylim(y_coords[0] - cell_size_y/2, y_coords[-1] + cell_size_y/2)

    for pos in led_positions:
        ax.plot(pos[0], pos[1], 'rx')

    add_rectangle(ax, (-37/2, -27/2), 37, 27, 'yellow')

    fig.canvas.draw()
    ax_w_px, ax_h_px = ax.get_window_extent(renderer=fig.canvas.get_renderer()).width, \
                       ax.get_window_extent(renderer=fig.canvas.get_renderer()).height
    scale_factor = min(ax_w_px/data_w, ax_h_px/data_h) * 1.2
    fontsize = max(6, scale_factor * 0.4 * 72 / fig.dpi)

    def draw_text(data):
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i,j]
                norm_val = (val - np.min(data)) / (np.ptp(data)+1e-9)
                color = 'white' if norm_val < 0.5 else 'black'
                t = ax.text(x_coords[i], y_coords[j], f"{val:.0f}",
                            ha='center', va='center', color=color, fontsize=fontsize)
                texts.append(t)
        return texts

    text_objects = draw_text(current_data['data'])
    today = datetime.today().strftime("%Y%m%d")

    # -------------------------------
    # SAVE FUNCTION
    # -------------------------------
    def save_current(name, data):
        """
        Saves PNG images for each dataset
        """
        # Update heatmap data
        heatmap.set_array(data.T.ravel())
        heatmap.set_clim(vmin=data.min(), vmax=data.max())
        cbar.update_normal(heatmap)

        # Update cell text
        for t in text_objects:
            t.remove()
        text_objects.clear()
        text_objects.extend(draw_text(data))

        # Maintain axes
        ax.set_xlim(x_coords[0] - cell_size_x/2, x_coords[-1] + cell_size_x/2)
        ax.set_ylim(y_coords[0] - cell_size_y/2, y_coords[-1] + cell_size_y/2)
        ax.set_aspect('equal', adjustable='box')

        # Save PNG
        filename = f"{name}_{output_dir.name.replace('output_','')}_{today}.png"
        full_path = output_dir / filename
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {full_path}")

    save_current("mean", E_mean)
    save_current("min", E_min)
    save_current("max", E_max)

    # -------------------------------
    # INTERACTIVE RADIOBUTTONS
    # -------------------------------
    axcolor = 'lightgoldenrodyellow'
    rax = plt.axes([0.01, 0.5 - 0.08/2, 0.12, 0.08], facecolor=axcolor)
    radio = RadioButtons(rax, ('E_mean','E_min','E_max'))

    def update(label):
        data = E_mean if label=='E_mean' else E_min if label=='E_min' else E_max
        current_data['data'] = data
        heatmap.set_array(data.T.ravel())
        heatmap.set_clim(vmin=data.min(), vmax=data.max())
        cbar.update_normal(heatmap)
        for t in text_objects:
            t.remove()
        text_objects.clear()
        text_objects.extend(draw_text(data))
        fig.canvas.draw_idle()

    radio.on_clicked(update)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def add_rectangle(ax, position, width, height, color):
    rect = plt.Rectangle(position, width, height, edgecolor=color, facecolor='none', lw=2)
    ax.add_patch(rect)