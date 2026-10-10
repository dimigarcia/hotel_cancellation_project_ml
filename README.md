# Hotel Booking Cancellation Prediction

A machine learning model that predicts, at booking time, which hotel bookings are likely to be cancelled. Machine Learning case study for the Master in Data Science & AI at Evolve Academy.

The tuned LightGBM model reaches a test AUC of 0.84. At the chosen threshold it catches 74% of cancellations with 67% precision. The notebook covers EDA, data leakage, feature engineering, a temporal train/validation/test split, model comparison, Optuna tuning and SHAP interpretability.

**Project page:** [dimigarcia.github.io/hotel_cancellation_project_ml](https://dimigarcia.github.io/hotel_cancellation_project_ml/) presents the results, key findings and a glossary of the technical terms.

## Contents
| File | Description |
|---|---|
| `hotel_cancellation_prediction.ipynb` | The analysis notebook, executed, with all outputs |
| `reservas_hoteleras.csv` | Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019), 119,390 bookings |
| `requirements.txt` | Pinned Python dependencies |
| `docs/index.html` | Source of the project page (GitHub Pages) |

## Setup (macOS)
Requires Python 3.11. SHAP depends on numba, which does not support Python 3.14 yet. The steps below use [uv](https://docs.astral.sh/uv/), which downloads Python 3.11 automatically.

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Then open a new terminal so `uv` is on the PATH.

2. From the project folder, create the environment and install the dependencies:
   ```bash
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. Register the environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name hotel-ml --display-name "Python (hotel-ml)"
   ```

4. Start JupyterLab with `jupyter lab`, or open the folder in VS Code. Select the **Python (hotel-ml)** kernel and run all cells. A full run takes about 2–3 minutes. The CSV must be in the same folder as the notebook.

## Troubleshooting
**`Library not loaded: @rpath/libomp.dylib` when importing LightGBM.** LightGBM needs the OpenMP runtime, which macOS does not include. Run the following from the project folder, with the environment active, to link the copy bundled with scikit-learn:
```bash
ln -s "$PWD/.venv/lib/python3.11/site-packages/sklearn/.dylibs/libomp.dylib" \
      "$(python -c 'import sys; print(sys.base_prefix)')/lib/libomp.dylib"
```
Then restart the kernel. `brew install libomp` also works.

**uv tries to build `llvmlite` from source on an Intel Mac.** Newer numpy releases have no compatible numba build for Intel Macs. `requirements.txt` pins `numpy==2.3.5` to avoid this.

## Uninstall
```bash
jupyter kernelspec uninstall hotel-ml
rm -rf .venv
```
