"""
execute_notebook.py
Executes burn_in_anomaly_and_drift_pipeline.ipynb sequentially in headless mode,
capturing stdout and matplotlib figures directly into the .ipynb JSON,
so that the notebook opens with complete, rich visualizations and outputs.
"""

import os
import sys
import io
import json
import base64
import contextlib

# Ensure non-blocking headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def execute_and_populate_notebook():
    nb_path = os.path.join(os.path.dirname(__file__), "burn_in_anomaly_and_drift_pipeline.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    orig_cwd = os.getcwd()
    notebook_dir = os.path.abspath(os.path.dirname(nb_path))
    os.chdir(notebook_dir)

    global_ns = {
        "__name__": "__main__",
        "__file__": nb_path
    }

    print("[EXECUTION] Starting notebook execution...", flush=True)
    cell_idx = 0
    execution_count = 1

    # Override plt.show to avoid blocking
    plt.show = lambda *args, **kwargs: None

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell_idx += 1
            code_str = "".join(cell["source"])
            outputs = []
            
            stdout_buf = io.StringIO()
            plt.close("all")

            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                    exec(code_str, global_ns)
                
                # Check for printed text output
                text_output = stdout_buf.getvalue()
                if text_output:
                    outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in text_output.splitlines()]
                    })

                # Check if matplotlib created any figures
                fig_nums = plt.get_fignums()
                for f_num in fig_nums:
                    fig = plt.figure(f_num)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=120)
                    img_buf.seek(0)
                    b64_data = base64.b64encode(img_buf.read()).decode("utf-8")
                    outputs.append({
                        "data": {
                            "image/png": b64_data,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                    plt.close(fig)

                cell["outputs"] = outputs
                cell["execution_count"] = execution_count
                execution_count += 1
                print(f"  [Cell {cell_idx}] Executed successfully.", flush=True)

            except Exception as e:
                print(f"  [Cell {cell_idx}] ERROR: {e}", flush=True)
                err_text = stdout_buf.getvalue() + f"\nTraceback error: {e}"
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [line + "\n" for line in err_text.splitlines()]
                })
                cell["outputs"] = outputs
                cell["execution_count"] = execution_count
                execution_count += 1
                os.chdir(orig_cwd)
                raise e

    os.chdir(orig_cwd)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"\n[SUCCESS] All {cell_idx} code cells executed cleanly and outputs embedded into {nb_path}!", flush=True)

if __name__ == "__main__":
    execute_and_populate_notebook()
