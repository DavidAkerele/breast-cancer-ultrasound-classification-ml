import json
import sys
import os
import io
import contextlib
import base64
import traceback

# Enforce headless matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def execute_and_populate_notebook(nb_path):
    print(f"[*] Starting execution of {nb_path}...", flush=True)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_scope = {"__name__": "__main__"}
    exec_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source_code = "".join(cell["source"])
            print(f"--- Executing Code Cell {idx} ---", flush=True)
            
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            outputs = []
            
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(source_code, global_scope)
                
                stdout_text = stdout_buf.getvalue()
                stderr_text = stderr_buf.getvalue()
                
                if stdout_text:
                    outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in stdout_text.splitlines()]
                    })
                if stderr_text:
                    outputs.append({
                        "name": "stderr",
                        "output_type": "stream",
                        "text": [line + "\n" for line in stderr_text.splitlines()]
                    })
                    
                # Save figures
                figs = [plt.figure(n) for n in plt.get_fignums()]
                for fig in figs:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    outputs.append({
                        "data": {
                            "image/png": img_base64,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                    plt.close(fig)
                
                # Check for pandas DataFrame / Styler output on last line
                last_line = [l.strip() for l in source_code.splitlines() if l.strip() and not l.strip().startswith("#")][-1]
                if "styled_table" in last_line or "df_table" in last_line:
                    try:
                        obj = eval(last_line, global_scope)
                        if hasattr(obj, "to_html"):
                            html_data = obj.to_html()
                            outputs.append({
                                "data": {
                                    "text/html": [html_data],
                                    "text/plain": [str(obj)]
                                },
                                "metadata": {},
                                "output_type": "execute_result",
                                "execution_count": exec_count
                            })
                    except Exception as e:
                        pass
                
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"[!] Error in cell {idx}: {e}", flush=True)
                outputs.append({
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": [line + "\n" for line in err_msg.splitlines()]
                })
            
            cell["outputs"] = outputs
            cell["execution_count"] = exec_count
            exec_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[*] Successfully executed and populated notebook: {nb_path}", flush=True)

if __name__ == "__main__":
    execute_and_populate_notebook("Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb")
