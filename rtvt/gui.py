"""Desktop pop-up GUI for the Roman Target Visibility Tool."""

from __future__ import annotations

import argparse
import os
import tempfile
from argparse import Namespace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np


def _prepare_gui_environment() -> None:
    cache_root = Path(tempfile.gettempdir()) / "rtvt-matplotlib"
    xdg_cache_root = Path(tempfile.gettempdir()) / "rtvt-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    xdg_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_root))


_prepare_gui_environment()

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if TYPE_CHECKING:
    from rtvt.cli import VisibilityResult


class RTVTGui(tk.Tk):
    """Small desktop GUI for local RTVT demos."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Roman Target Visibility Tool")
        self.geometry("1180x760")
        self.minsize(980, 650)

        self.result: VisibilityResult | None = None
        self.target_name_var = tk.StringVar(value="Demo target")
        self.coordinate_system_var = tk.StringVar(value="equatorial")
        self.ra_var = tk.StringVar(value="90.0")
        self.dec_var = tk.StringVar(value="-1.0")
        self.start_date_var = tk.StringVar(value="2024-01-01")
        self.duration_var = tk.StringVar(value="365")
        self.sampling_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Ready")

        self.figure = None
        self.canvas: FigureCanvasTkAgg | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=(14, 14))
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="RTVT", font=("TkDefaultFont", 20, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Label(controls, text="Roman Target Visibility Tool").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 18)
        )

        fields = [
            ("Target name", self.target_name_var),
            ("Longitude", self.ra_var),
            ("Latitude", self.dec_var),
            ("Start date", self.start_date_var),
            ("Duration days", self.duration_var),
            ("Sampling days", self.sampling_var),
        ]
        ttk.Label(controls, text="Coordinates").grid(row=2, column=0, sticky="w", pady=5)
        coord_box = ttk.Combobox(
            controls,
            textvariable=self.coordinate_system_var,
            values=("equatorial", "galactic"),
            state="readonly",
            width=21,
        )
        coord_box.grid(row=2, column=1, sticky="ew", pady=5, padx=(10, 0))
        coord_box.bind("<<ComboboxSelected>>", lambda _event: self._update_coordinate_labels())

        self.coord_label_widgets = {}
        for row, (label, var) in enumerate(fields, start=3):
            label_widget = ttk.Label(controls, text=label)
            label_widget.grid(row=row, column=0, sticky="w", pady=5)
            if label in {"Longitude", "Latitude"}:
                self.coord_label_widgets[label] = label_widget
            ttk.Entry(controls, textvariable=var, width=24).grid(
                row=row, column=1, sticky="ew", pady=5, padx=(10, 0)
            )

        ttk.Button(controls, text="Compute Visibility", command=self.compute).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(16, 6)
        )
        ttk.Button(controls, text="Save CSV", command=self.save_csv).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=4
        )
        ttk.Button(controls, text="Save Plot", command=self.save_plot).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=4
        )
        ttk.Button(controls, text="Copy CLI Command", command=self.copy_cli_command).grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=4
        )

        self.coordinate_help_label = ttk.Label(
            controls,
            wraplength=260,
            foreground="#4f5b66",
        )
        self.coordinate_help_label.grid(row=13, column=0, columnspan=2, sticky="w", pady=(18, 0))
        self._update_coordinate_labels()

        main = ttk.Frame(self, padding=(0, 14, 14, 14))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        metrics = ttk.Frame(main)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            metrics.columnconfigure(col, weight=1)

        self.metric_vars = {
            "visible_samples": tk.StringVar(value="-"),
            "visible_fraction": tk.StringVar(value="-"),
            "windows": tk.StringVar(value="-"),
            "sampling": tk.StringVar(value="-"),
        }
        metric_specs = [
            ("Visible samples", "visible_samples"),
            ("Visible fraction", "visible_fraction"),
            ("Observable windows", "windows"),
            ("Sampling", "sampling"),
        ]
        for col, (label, key) in enumerate(metric_specs):
            card = ttk.LabelFrame(metrics, text=label, padding=(10, 8))
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
            ttk.Label(card, textvariable=self.metric_vars[key], font=("TkDefaultFont", 16, "bold")).pack(anchor="w")

        notebook = ttk.Notebook(main)
        notebook.grid(row=1, column=0, sticky="nsew")

        self.plot_frame = ttk.Frame(notebook)
        self.summary_frame = ttk.Frame(notebook)
        self.windows_frame = ttk.Frame(notebook)
        self.data_frame = ttk.Frame(notebook)

        notebook.add(self.plot_frame, text="Plot")
        notebook.add(self.summary_frame, text="Summary")
        notebook.add(self.windows_frame, text="Windows")
        notebook.add(self.data_frame, text="Sampled Data")

        self._build_plot_tab()
        self._build_summary_tab()
        self._build_windows_tab()
        self._build_data_tab()

        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 4)).grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )

    def _build_plot_tab(self) -> None:
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_placeholder = ttk.Label(
            self.plot_frame,
            text="Compute a target to display the visibility plot.",
            anchor="center",
        )
        self.plot_placeholder.grid(row=0, column=0, sticky="nsew")

    def _build_summary_tab(self) -> None:
        self.summary_frame.columnconfigure(0, weight=1)
        self.summary_frame.rowconfigure(0, weight=1)
        self.summary_text = tk.Text(self.summary_frame, wrap="word", height=18)
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.summary_frame, command=self.summary_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=scroll.set)

    def _build_windows_tab(self) -> None:
        self.windows_frame.columnconfigure(0, weight=1)
        self.windows_frame.rowconfigure(0, weight=1)
        columns = ("window_start", "window_end", "duration_days", "nominal_roll_start", "nominal_roll_end")
        self.windows_tree = ttk.Treeview(self.windows_frame, columns=columns, show="headings")
        headings = {
            "window_start": "Window Start",
            "window_end": "Window End",
            "duration_days": "Duration (days)",
            "nominal_roll_start": "Roll Start",
            "nominal_roll_end": "Roll End",
        }
        for column in columns:
            self.windows_tree.heading(column, text=headings[column])
            self.windows_tree.column(column, width=150, anchor="w")
        self.windows_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.windows_frame, command=self.windows_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.windows_tree.configure(yscrollcommand=scroll.set)

    def _build_data_tab(self) -> None:
        self.data_frame.columnconfigure(0, weight=1)
        self.data_frame.rowconfigure(0, weight=1)
        columns = ("time_isot", "separation", "good_angles", "nominal_roll", "pa_obs_y")
        self.data_tree = ttk.Treeview(self.data_frame, columns=columns, show="headings")
        headings = {
            "time_isot": "Time",
            "separation": "Separation",
            "good_angles": "In FOR",
            "nominal_roll": "Nominal Roll",
            "pa_obs_y": "PA Obs Y",
        }
        for column in columns:
            self.data_tree.heading(column, text=headings[column])
            self.data_tree.column(column, width=145, anchor="w")
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.data_frame, command=self.data_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.data_tree.configure(yscrollcommand=scroll.set)

    def _update_coordinate_labels(self) -> None:
        if self.coordinate_system_var.get() == "galactic":
            self.coord_label_widgets["Longitude"].configure(text="Galactic l")
            self.coord_label_widgets["Latitude"].configure(text="Galactic b")
            self.coordinate_help_label.configure(
                text="Galactic l and b are entered in decimal degrees."
            )
        else:
            self.coord_label_widgets["Longitude"].configure(text="RA")
            self.coord_label_widgets["Latitude"].configure(text="Dec")
            self.coordinate_help_label.configure(
                text=(
                    "RA accepts decimal degrees or sexagesimal hour angle.\n"
                    "Dec accepts decimal degrees or sexagesimal degrees."
                )
            )

    def compute(self) -> None:
        try:
            duration_days = float(self.duration_var.get())
            sampling_days = float(self.sampling_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Duration and sampling days must be numeric.")
            return

        if duration_days <= 0 or sampling_days <= 0:
            messagebox.showerror("Invalid input", "Duration and sampling days must be positive.")
            return
        if sampling_days > duration_days:
            messagebox.showerror("Invalid input", "Sampling days cannot exceed duration days.")
            return

        self.status_var.set("Computing visibility...")
        self.update_idletasks()

        args = Namespace(
            ra=self.ra_var.get().strip(),
            dec=self.dec_var.get().strip(),
            coordinate_system=self.coordinate_system_var.get(),
            start_date=self.start_date_var.get().strip(),
            duration_days=duration_days,
            sampling_days=sampling_days,
            target_name=self.target_name_var.get().strip(),
            write_csv=None,
            write_plot=None,
            show_plot=False,
            quiet=True,
        )

        try:
            from rtvt.cli import run_visibility

            self.result = run_visibility(args)
        except Exception as exc:
            self.status_var.set("Ready")
            messagebox.showerror("Computation failed", str(exc))
            return

        self._render_result()
        self.status_var.set("Visibility computed.")

    def _render_result(self) -> None:
        if self.result is None:
            return

        from rtvt.cli import summarize_windows

        target_name = self.target_name_var.get().strip() or self.result.label
        good = self.result.table["good_angles"].astype(bool).to_numpy()
        windows = summarize_windows(self.result)
        vis_fraction = float(np.mean(good)) if len(good) else 0.0

        self.metric_vars["visible_samples"].set(f"{int(np.sum(good))}/{len(good)}")
        self.metric_vars["visible_fraction"].set(f"{vis_fraction * 100:.1f}%")
        self.metric_vars["windows"].set(str(len(windows)))
        self.metric_vars["sampling"].set(f"{self.result.sampling_days:g} day(s)")

        self._render_summary(target_name)
        self._render_windows(windows)
        self._render_sampled_data()
        self._render_plot(target_name)

    def _render_summary(self, target_name: str) -> None:
        from rtvt.cli import format_summary

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, format_summary(self.result, target_name=target_name))
        self.summary_text.configure(state="normal")

    def _render_windows(self, windows) -> None:
        self.windows_tree.delete(*self.windows_tree.get_children())
        for _, row in windows.iterrows():
            self.windows_tree.insert(
                "",
                tk.END,
                values=(
                    row["window_start"],
                    row["window_end"],
                    f"{row['duration_days']:.1f}",
                    f"{row['nominal_roll_start']:.3f}",
                    f"{row['nominal_roll_end']:.3f}",
                ),
            )

    def _render_sampled_data(self) -> None:
        self.data_tree.delete(*self.data_tree.get_children())
        table = self.result.table
        times = [time.isot for time in self.result.sampled_times]
        for idx, (_, row) in enumerate(table.head(500).iterrows()):
            self.data_tree.insert(
                "",
                tk.END,
                values=(
                    times[idx],
                    f"{float(row['separation']):.3f}",
                    str(bool(row["good_angles"])),
                    f"{float(row['nominal_roll']):.3f}",
                    f"{float(row['pa_obs_y']):.3f}",
                ),
            )

    def _render_plot(self, target_name: str) -> None:
        from rtvt.cli import make_plot

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.plot_placeholder.winfo_ismapped():
            self.plot_placeholder.grid_remove()

        self.figure = make_plot(self.result, target_name=target_name)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def save_csv(self) -> None:
        if self.result is None:
            messagebox.showinfo("No result", "Compute visibility before saving CSV.")
            return
        path = filedialog.asksaveasfilename(
            title="Save RTVT CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="visibility.csv",
        )
        if not path:
            return
        from rtvt.cli import write_csv

        write_csv(self.result, path)
        self.status_var.set(f"Saved CSV: {Path(path).name}")

    def save_plot(self) -> None:
        if self.result is None or self.figure is None:
            messagebox.showinfo("No plot", "Compute visibility before saving a plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save RTVT plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile="visibility.png",
        )
        if not path:
            return
        from rtvt.cli import save_plot_figure

        save_plot_figure(self.figure, path)
        self.status_var.set(f"Saved plot: {Path(path).name}")

    def copy_cli_command(self) -> None:
        duration = self.duration_var.get().strip()
        sampling = self.sampling_var.get().strip()
        if self.coordinate_system_var.get() == "galactic":
            lon_arg, lat_arg = "--lon", "--lat"
        else:
            lon_arg, lat_arg = "--ra", "--dec"
        command = (
            "rtvt "
            f"{lon_arg} {self.ra_var.get().strip()} "
            f"{lat_arg} {self.dec_var.get().strip()} "
            f"--coordinate-system {self.coordinate_system_var.get()} "
            f"--start-date {self.start_date_var.get().strip()} "
            f"--duration-days {duration} "
            f"--sampling-days {sampling} "
            "--write-csv visibility.csv "
            "--write-plot visibility.png"
        )
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Copied CLI command to clipboard.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtvt-gui",
        description="Open the RTVT desktop GUI.",
    )
    parser.parse_args(argv)

    app = RTVTGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
