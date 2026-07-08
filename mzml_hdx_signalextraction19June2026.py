import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pyteomics import mzml, mass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import csv
import re

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 24
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 24
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 8

ADDUCT_MASSES = {
    "H": 1.007276466812,
    "Na": 22.9897692820,
    "K": 38.9637064864,
}

H_MASS = 1.007825
D_MASS = 2.0141017778
D_SHIFT = D_MASS - H_MASS


class IsotopePeakFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("HDX Isotope Peak Finder")
        self.root.geometry("1350x950")

        self.file_path = None
        self.results = []
        self.matrix_rows = []
        self.tic_data = None
        self.max_isotopes = 6
        self.adduct_states = []
        self.state_filter_vars = {}

        ttk.Label(root, text="Amino Acid Sequence OR Neutral Monoisotopic Mass:").pack(pady=5)

        input_frame = ttk.Frame(root)
        input_frame.pack(pady=5)

        ttk.Label(input_frame, text="Sequence:").grid(row=0, column=0, padx=5)
        self.seq_entry = ttk.Entry(input_frame, width=30)
        self.seq_entry.grid(row=0, column=1, padx=5)

        ttk.Label(input_frame, text="Mass:").grid(row=0, column=2, padx=5)
        self.mass_entry = ttk.Entry(input_frame, width=15)
        self.mass_entry.grid(row=0, column=3, padx=5)

        state_frame = ttk.LabelFrame(root, text="Adduct + charge states")
        state_frame.pack(pady=8, fill=tk.X, padx=10)

        ttk.Label(state_frame, text="Charge:").grid(row=0, column=0, padx=5)
        self.state_charge_entry = ttk.Entry(state_frame, width=8)
        self.state_charge_entry.insert(0, "2")
        self.state_charge_entry.grid(row=0, column=1, padx=5)

        ttk.Label(state_frame, text="Adduct formula, e.g. 2H, Na+H, 2Na:").grid(row=0, column=2, padx=5)
        self.state_adduct_entry = ttk.Entry(state_frame, width=20)
        self.state_adduct_entry.insert(0, "2H")
        self.state_adduct_entry.grid(row=0, column=3, padx=5)

        ttk.Button(state_frame, text="Add State", command=self.add_adduct_state).grid(row=0, column=4, padx=5)
        ttk.Button(state_frame, text="Remove Selected", command=self.remove_selected_adduct_state).grid(row=0, column=5, padx=5)

        self.state_listbox = tk.Listbox(state_frame, height=4, width=70)
        self.state_listbox.grid(row=1, column=0, columnspan=6, pady=5, padx=5, sticky="w")

        settings_frame = ttk.Frame(root)
        settings_frame.pack(pady=5)

        ttk.Label(settings_frame, text="PPM tolerance:").grid(row=0, column=0, padx=5)
        self.ppm_entry = ttk.Entry(settings_frame, width=10)
        self.ppm_entry.insert(0, "10")
        self.ppm_entry.grid(row=0, column=1, padx=5)

        ttk.Label(settings_frame, text="Number of D peaks:").grid(row=0, column=2, padx=5)
        self.isotope_entry = ttk.Entry(settings_frame, width=10)
        self.isotope_entry.insert(0, "6")
        self.isotope_entry.grid(row=0, column=3, padx=5)

        ttk.Label(settings_frame, text="Minimum intensity:").grid(row=0, column=4, padx=5)
        self.min_intensity_entry = ttk.Entry(settings_frame, width=10)
        self.min_intensity_entry.insert(0, "100")
        self.min_intensity_entry.grid(row=0, column=5, padx=5)

        ttk.Button(root, text="Load mzML File", command=self.load_file).pack(pady=5)

        self.file_label = ttk.Label(root, text="No file selected")
        self.file_label.pack()

        scan_frame = ttk.Frame(root)
        scan_frame.pack(pady=5)

        ttk.Label(scan_frame, text="Scan ranges to process, e.g. 1-10,15-20:").pack(side=tk.LEFT)
        self.scan_entry = ttk.Entry(scan_frame, width=25)
        self.scan_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(root, text="Find HDX Peaks Scan-by-Scan", command=self.find_peaks).pack(pady=8)
        ttk.Button(root, text="Export Matrix CSV", command=self.export_matrix_csv).pack(pady=3)

        self.filter_frame = ttk.LabelFrame(root, text="States shown in graphs")
        self.filter_frame.pack(pady=8, fill=tk.X, padx=10)

        ttk.Label(self.filter_frame, text="Run analysis to generate graph filters.").pack(pady=5)

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Select All States", command=self.select_all_states).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Deselect All States", command=self.deselect_all_states).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Plot Normalised Isotope Intensities 3D", command=self.plot_isotope_intensities).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Plot Average Deuterium Uptake", command=self.plot_deuterium_uptake).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Plot Maximum Deuterium Uptake", command=self.plot_max_deuterium_uptake).grid(row=0, column=4, padx=5)

        export_frame = ttk.Frame(root)
        export_frame.pack(pady=5)

        ttk.Button(export_frame, text="Export Isotope Plot Data", command=self.export_isotope_plot_data).grid(row=0, column=0, padx=5)
        ttk.Button(export_frame, text="Export Uptake Plot Data", command=self.export_uptake_plot_data).grid(row=0, column=1, padx=5)
        ttk.Button(export_frame, text="Export Max Uptake Plot Data", command=self.export_max_uptake_plot_data).grid(row=0, column=2, padx=5)

        columns = (
            "Scan", "State", "Charge", "Adduct", "D#",
            "Theoretical m/z", "Found m/z", "Raw Intensity", "Normalised %"
        )

        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=12)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=130)

        self.tree.pack(pady=5, fill=tk.BOTH, expand=True)

        self.add_default_states()

    def add_default_states(self):
        defaults = [
            (1, "H"),
            (2, "2H"),
            (2, "Na+H"),
            (2, "2Na"),
            (3, "3H"),
            (3, "Na+2H"),
        ]

        for charge, formula in defaults:
            self.add_adduct_state_direct(charge, formula)

    def parse_adduct_formula(self, formula):
        formula = formula.replace(" ", "")
        parts = formula.split("+")

        total_mass = 0.0

        for part in parts:
            match = re.fullmatch(r"(\d*)([A-Za-z]+)", part)

            if not match:
                raise ValueError(f"Invalid adduct part: {part}")

            count_text, species = match.groups()
            count = int(count_text) if count_text else 1

            if species not in ADDUCT_MASSES:
                raise ValueError(f"Unknown adduct: {species}")

            total_mass += count * ADDUCT_MASSES[species]

        return total_mass

    def make_state_label(self, charge, formula):
        return f"[M+{formula}]{charge}+"

    def add_adduct_state_direct(self, charge, formula):
        try:
            adduct_mass = self.parse_adduct_formula(formula)
        except ValueError:
            return

        label = self.make_state_label(charge, formula)

        for state in self.adduct_states:
            if state["label"] == label:
                return

        self.adduct_states.append({
            "charge": charge,
            "formula": formula,
            "mass": adduct_mass,
            "label": label
        })

        self.refresh_state_listbox()

    def add_adduct_state(self):
        try:
            charge = int(self.state_charge_entry.get())
            formula = self.state_adduct_entry.get().strip()

            if charge <= 0:
                raise ValueError("Charge must be positive.")

            adduct_mass = self.parse_adduct_formula(formula)

        except Exception as e:
            messagebox.showerror("Invalid State", str(e))
            return

        label = self.make_state_label(charge, formula)

        for state in self.adduct_states:
            if state["label"] == label:
                messagebox.showwarning("Duplicate", "This state already exists.")
                return

        self.adduct_states.append({
            "charge": charge,
            "formula": formula,
            "mass": adduct_mass,
            "label": label
        })

        self.refresh_state_listbox()

    def remove_selected_adduct_state(self):
        selection = self.state_listbox.curselection()

        if not selection:
            return

        idx = selection[0]
        del self.adduct_states[idx]
        self.refresh_state_listbox()

    def refresh_state_listbox(self):
        self.state_listbox.delete(0, tk.END)

        for state in self.adduct_states:
            self.state_listbox.insert(
                tk.END,
                f"{state['label']}   adduct mass={state['mass']:.8f}"
            )

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("mzML files", "*.mzML")])

        if path:
            self.file_path = path
            self.file_label.config(text=f"Loaded: {path}")
            self.load_spectrum_and_plot_tic()

    def load_spectrum_and_plot_tic(self):
        tic_values = []
        scan_numbers = []
        spectra = []

        try:
            with mzml.read(self.file_path) as reader:
                for i, spec in enumerate(reader):
                    if spec.get("ms level", 1) == 1:
                        mzs = np.atleast_1d(np.array(spec.get("m/z array", []), dtype=float))
                        ints = np.atleast_1d(np.array(spec.get("intensity array", []), dtype=float))

                        tic_values.append(np.nansum(ints))
                        scan_numbers.append(i + 1)
                        spectra.append((mzs, ints))

            self.tic_data = (scan_numbers, tic_values, spectra)

            fig, ax = plt.subplots(figsize=(16, 10))
            ax.plot(scan_numbers, tic_values, lw=2)
            ax.set_xlabel("Scan Number")
            ax.set_ylabel("Relative Intensity")
            ax.set_title("Total Ion Current")

            plot_window = tk.Toplevel(self.root)
            plot_window.title("TIC vs Scan Number")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read mzML file:\n{e}")

    def parse_scan_ranges(self, text):
        ranges = []

        try:
            for part in text.split(","):
                part = part.strip()

                if "-" in part:
                    start, end = map(int, part.split("-"))
                else:
                    start = end = int(part)

                ranges.append((start - 1, end - 1))

        except Exception:
            messagebox.showerror(
                "Invalid Input",
                "Enter ranges like 1-10,15-20 or single scans like 5,8,10."
            )

        return ranges

    def find_peaks(self):
        seq = self.seq_entry.get().strip()
        mass_text = self.mass_entry.get().strip()

        if not self.adduct_states:
            messagebox.showerror("Error", "Please add at least one adduct/charge state.")
            return

        if not seq and not mass_text:
            messagebox.showerror("Error", "Please enter either a sequence or a neutral monoisotopic mass.")
            return

        if seq and mass_text:
            messagebox.showerror("Error", "Please enter either a sequence OR a mass, not both.")
            return

        if not self.file_path or self.tic_data is None:
            messagebox.showerror("Error", "Please load an mzML file first.")
            return

        try:
            ppm_tol = float(self.ppm_entry.get())
            max_isotopes = int(self.isotope_entry.get())
            min_intensity = float(self.min_intensity_entry.get())

            if max_isotopes <= 0:
                raise ValueError

        except ValueError:
            messagebox.showerror("Error", "PPM, D peaks, and minimum intensity must be numeric.")
            return

        try:
            mono_mass = mass.calculate_mass(sequence=seq) if seq else float(mass_text)
        except Exception as e:
            messagebox.showerror("Error", f"Could not calculate or read mass:\n{e}")
            return

        scan_text = self.scan_entry.get().strip()

        if not scan_text:
            messagebox.showwarning("No Scans", "Please specify scan ranges first.")
            return

        ranges = self.parse_scan_ranges(scan_text)
        spectra = self.tic_data[2]

        results = []
        matrix_dict = {}

        for start, end in ranges:
            for scan_idx in range(start, end + 1):

                if scan_idx < 0 or scan_idx >= len(spectra):
                    continue

                mzs, ints = spectra[scan_idx]
                scan_no = scan_idx + 1

                if mzs.size == 0:
                    continue

                for state in self.adduct_states:
                    z = state["charge"]
                    adduct_formula = state["formula"]
                    adduct_mass = state["mass"]
                    state_label = state["label"]

                    row_key = (scan_no, state_label)

                    matrix_dict[row_key] = {
                        "Scan": scan_no,
                        "State": state_label,
                        "Charge": z,
                        "Adduct": adduct_formula,
                        "Adduct_mass": adduct_mass
                    }

                    base_mz = (mono_mass + adduct_mass) / z

                    for d_number in range(max_isotopes):
                        theo_mz = base_mz + (d_number * D_SHIFT) / z
                        delta = theo_mz * ppm_tol / 1e6

                        mask = (
                            (mzs >= theo_mz - delta)
                            & (mzs <= theo_mz + delta)
                        )

                        intensity_value = 0
                        found_mz_value = ""

                        if np.any(mask):
                            local_mz = mzs[mask]
                            local_int = ints[mask]

                            if local_int.size > 0:
                                peak_idx = np.argmax(local_int)
                                candidate_mz = float(local_mz[peak_idx])
                                candidate_intensity = float(local_int[peak_idx])

                                if candidate_intensity >= min_intensity:
                                    found_mz_value = candidate_mz
                                    intensity_value = candidate_intensity

                        matrix_dict[row_key][f"D{d_number}_theoretical_mz"] = float(theo_mz)
                        matrix_dict[row_key][f"D{d_number}_found_mz"] = found_mz_value
                        matrix_dict[row_key][f"D{d_number}_raw_intensity"] = intensity_value

                        if intensity_value > 0:
                            results.append([
                                scan_no,
                                state_label,
                                z,
                                adduct_formula,
                                d_number,
                                float(theo_mz),
                                found_mz_value,
                                intensity_value
                            ])
                        else:
                            break

        self.results = results
        self.matrix_rows = list(matrix_dict.values())
        self.max_isotopes = max_isotopes

        for row in self.matrix_rows:
            max_intensity = 0

            for d_number in range(max_isotopes):
                intensity = row.get(f"D{d_number}_raw_intensity", 0)
                max_intensity = max(max_intensity, intensity)

            for d_number in range(max_isotopes):
                intensity = row.get(f"D{d_number}_raw_intensity", 0)

                if max_intensity > 0:
                    norm_percent = (intensity / max_intensity) * 100
                else:
                    norm_percent = 0

                row[f"D{d_number}_normalised_percent"] = norm_percent

        for r in self.tree.get_children():
            self.tree.delete(r)

        for r in results:
            scan_no, state_label, charge, adduct, d_number, theo_mz, found_mz, raw_intensity = r

            norm_percent = 0

            for row in self.matrix_rows:
                if row["Scan"] == scan_no and row["State"] == state_label:
                    norm_percent = row.get(f"D{d_number}_normalised_percent", 0)
                    break

            self.tree.insert(
                "",
                tk.END,
                values=[
                    scan_no,
                    state_label,
                    charge,
                    adduct,
                    d_number,
                    f"{theo_mz:.8f}",
                    f"{found_mz:.8f}",
                    f"{raw_intensity:.1f}",
                    f"{norm_percent:.1f}"
                ]
            )

        self.update_state_filters()

        messagebox.showinfo("Done", f"Found {len(results)} HDX peaks across scans.")

    def update_state_filters(self):
        for child in self.filter_frame.winfo_children():
            child.destroy()

        self.state_filter_vars = {}

        states = sorted({row["State"] for row in self.matrix_rows})

        if not states:
            ttk.Label(self.filter_frame, text="No states found.").pack(pady=5)
            return

        for idx, state in enumerate(states):
            var = tk.BooleanVar(value=True)
            self.state_filter_vars[state] = var

            cb = ttk.Checkbutton(
                self.filter_frame,
                text=state,
                variable=var
            )

            cb.grid(row=idx // 6, column=idx % 6, padx=8, pady=4, sticky="w")

    def select_all_states(self):
        for var in self.state_filter_vars.values():
            var.set(True)

    def deselect_all_states(self):
        for var in self.state_filter_vars.values():
            var.set(False)

    def selected_states(self):
        return {
            state
            for state, var in self.state_filter_vars.items()
            if var.get()
        }

    def export_matrix_csv(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Matrix", "Run the scan-by-scan analysis first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not path:
            return

        fieldnames = ["Scan", "State", "Charge", "Adduct", "Adduct_mass"]

        for d_number in range(self.max_isotopes):
            fieldnames.extend([
                f"D{d_number}_theoretical_mz",
                f"D{d_number}_found_mz",
                f"D{d_number}_raw_intensity",
                f"D{d_number}_normalised_percent"
            ])

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.matrix_rows)

        messagebox.showinfo("Exported", f"Matrix saved to:\n{path}")

    def get_isotope_plot_data(self, filtered=True):
        rows = []
        selected = self.selected_states() if filtered else None

        for row in self.matrix_rows:
            if filtered and row["State"] not in selected:
                continue

            for d_number in range(self.max_isotopes):
                rows.append({
                    "Scan": row["Scan"],
                    "State": row["State"],
                    "Charge": row["Charge"],
                    "Adduct": row["Adduct"],
                    "D_number": d_number,
                    "Theoretical_mz": row.get(f"D{d_number}_theoretical_mz", ""),
                    "Found_mz": row.get(f"D{d_number}_found_mz", ""),
                    "Raw_intensity": row.get(f"D{d_number}_raw_intensity", 0),
                    "Normalised_percent": row.get(f"D{d_number}_normalised_percent", 0)
                })

        return rows

    def get_uptake_plot_data(self, filtered=True):
        rows = []
        selected = self.selected_states() if filtered else None

        for row in self.matrix_rows:
            if filtered and row["State"] not in selected:
                continue

            weighted_sum = 0
            total_intensity = 0

            for d_number in range(self.max_isotopes):
                intensity = row.get(f"D{d_number}_raw_intensity", 0)
                weighted_sum += d_number * intensity
                total_intensity += intensity

            uptake = weighted_sum / total_intensity if total_intensity > 0 else np.nan

            rows.append({
                "Scan": row["Scan"],
                "State": row["State"],
                "Charge": row["Charge"],
                "Adduct": row["Adduct"],
                "Average_deuterium_uptake": uptake,
                "Total_intensity": total_intensity
            })

        return rows

    def get_max_uptake_plot_data(self, filtered=True):
        rows = []
        selected = self.selected_states() if filtered else None

        for row in self.matrix_rows:
            if filtered and row["State"] not in selected:
                continue

            max_d = np.nan
            total_intensity = 0

            for d_number in range(self.max_isotopes):
                intensity = row.get(f"D{d_number}_raw_intensity", 0)

                if intensity > 0:
                    max_d = d_number
                    total_intensity += intensity

            rows.append({
                "Scan": row["Scan"],
                "State": row["State"],
                "Charge": row["Charge"],
                "Adduct": row["Adduct"],
                "Maximum_deuterium_uptake": max_d,
                "Total_intensity": total_intensity
            })

        return rows

    def shade_colour(self, base_colour, d_number):
        max_d = max(self.max_isotopes - 1, 1)
        fraction = d_number / max_d

        rgb = np.array(base_colour[:3])
        white = np.array([1.0, 1.0, 1.0])

        shade = rgb * (1 - 0.65 * fraction) + white * (0.65 * fraction)

        return shade

    def plot_isotope_intensities(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Data", "Run the scan-by-scan analysis first.")
            return

        if not self.selected_states():
            messagebox.showwarning("No States", "Select at least one state.")
            return

        data = self.get_isotope_plot_data(filtered=True)

        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111, projection="3d")

        selected_state_list = sorted(self.selected_states())
        cmap = plt.get_cmap("tab10")

        state_colours = {
            state: cmap(i % 10)
            for i, state in enumerate(selected_state_list)
        }

        for state in selected_state_list:
            state_data = [row for row in data if row["State"] == state]
            base_colour = state_colours[state]

            for d_number in range(self.max_isotopes):
                d_data = [row for row in state_data if row["D_number"] == d_number]

                if not d_data:
                    continue

                scans = [row["Scan"] for row in d_data]
                d_values = [d_number] * len(d_data)
                intensities = [row["Normalised_percent"] for row in d_data]

                colour = self.shade_colour(base_colour, d_number)

                ax.plot(
                    scans,
                    d_values,
                    intensities,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=colour,
                    label=f"{state}, D{d_number}"
                )

        ax.set_xlabel("Scan Number", labelpad=20)
        ax.set_ylabel("Deuterium Number", labelpad=20)
        ax.set_zlabel("Normalised Intensity (%)", labelpad=20)
        ax.set_title("Normalised Isotope Intensity Plot")
        ax.view_init(elev=25, azim=-55)

        ax.legend(
            fontsize=10,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.15, 0.5),
            borderaxespad=0
        )

        fig.tight_layout(rect=[0, 0, 0.72, 1])

        win = tk.Toplevel(self.root)
        win.title("Normalised Isotope Intensity 3D Plot")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_deuterium_uptake(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Data", "Run the scan-by-scan analysis first.")
            return

        if not self.selected_states():
            messagebox.showwarning("No States", "Select at least one state.")
            return

        data = self.get_uptake_plot_data(filtered=True)

        fig, ax = plt.subplots(figsize=(18, 12))
        groups = {}

        selected_state_list = sorted(self.selected_states())
        cmap = plt.get_cmap("tab10")
        state_colours = {
            state: cmap(i % 10)
            for i, state in enumerate(selected_state_list)
        }

        for row in data:
            key = row["State"]
            groups.setdefault(key, {"Scan": [], "Uptake": []})
            groups[key]["Scan"].append(row["Scan"])
            groups[key]["Uptake"].append(row["Average_deuterium_uptake"])

        lines = []

        for state, values in groups.items():
            line, = ax.plot(
                values["Scan"],
                values["Uptake"],
                marker="o",
                markersize=6,
                linewidth=3,
                color=state_colours[state],
                label=state
            )
            lines.append((line, state))

        ax.set_xlabel("Scan Number")
        ax.set_ylabel("Average Deuterium Uptake")
        ax.set_title("Average Deuterium Uptake")
        ax.grid(True, alpha=0.3)

        ax.legend(
            fontsize=14,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )

        fig.tight_layout(rect=[0, 0, 0.78, 1])

       

        win = tk.Toplevel(self.root)
        win.title("Average Deuterium Uptake")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_max_deuterium_uptake(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Data", "Run the scan-by-scan analysis first.")
            return

        if not self.selected_states():
            messagebox.showwarning("No States", "Select at least one state.")
            return

        data = self.get_max_uptake_plot_data(filtered=True)

        fig, ax = plt.subplots(figsize=(18, 12))
        groups = {}

        selected_state_list = sorted(self.selected_states())
        cmap = plt.get_cmap("tab10")
        state_colours = {
            state: cmap(i % 10)
            for i, state in enumerate(selected_state_list)
        }

        for row in data:
            key = row["State"]
            groups.setdefault(key, {"Scan": [], "MaxD": []})
            groups[key]["Scan"].append(row["Scan"])
            groups[key]["MaxD"].append(row["Maximum_deuterium_uptake"])

        lines = []

        for state, values in groups.items():
            line, = ax.plot(
                values["Scan"],
                values["MaxD"],
                marker="o",
                markersize=6,
                linewidth=3,
                color=state_colours[state],
                label=state
            )
            lines.append((line, state))

        ax.set_xlabel("Scan Number")
        ax.set_ylabel("Maximum Deuterium Uptake")
        ax.set_title("Maximum Deuterium Uptake")
        ax.grid(True, alpha=0.3)

        ax.legend(
            fontsize=14,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )

        fig.tight_layout(rect=[0, 0, 0.78, 1])




        win = tk.Toplevel(self.root)
        win.title("Maximum Deuterium Uptake")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_isotope_plot_data(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Data", "Run the scan-by-scan analysis first.")
            return

        data = self.get_isotope_plot_data(filtered=True)

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not path:
            return

        fieldnames = [
            "Scan", "State", "Charge", "Adduct", "D_number",
            "Theoretical_mz", "Found_mz", "Raw_intensity", "Normalised_percent"
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        messagebox.showinfo("Exported", f"Filtered isotope plot data saved to:\n{path}")

    def export_uptake_plot_data(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Data", "Run the scan-by-scan analysis first.")
            return

        data = self.get_uptake_plot_data(filtered=True)

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not path:
            return

        fieldnames = [
            "Scan", "State", "Charge", "Adduct",
            "Average_deuterium_uptake", "Total_intensity"
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        messagebox.showinfo("Exported", f"Filtered uptake plot data saved to:\n{path}")

    def export_max_uptake_plot_data(self):
        if not self.matrix_rows:
            messagebox.showwarning("No Data", "Run the scan-by-scan analysis first.")
            return

        data = self.get_max_uptake_plot_data(filtered=True)

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not path:
            return

        fieldnames = [
            "Scan", "State", "Charge", "Adduct",
            "Maximum_deuterium_uptake", "Total_intensity"
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        messagebox.showinfo("Exported", f"Filtered maximum uptake plot data saved to:\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = IsotopePeakFinder(root)
    root.mainloop()