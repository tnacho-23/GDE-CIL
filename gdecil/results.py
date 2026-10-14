"""Writing accuracy matrices."""
import os

import pandas as pd

from .paths import RESULTS_DIR


def results_path(backbone: str, method: str, dataset: str, tags=(), root: str = RESULTS_DIR) -> str:
    """results/{backbone}/{method}_{dataset}[_{tag}...].xlsx; tags describe non-default
    settings (e.g. 'pca130', 'eps0.0001') so different settings do not overwrite each other."""
    return os.path.join(root, backbone, '_'.join([method, dataset, *tags]) + '.xlsx')


def save_results(path: str, matrices, extra_sheets=None):
    """One sheet per run (Run_1, Run_2, ...) with rows After_Task_k and columns
    Task_j; `extra_sheets` maps sheet names to DataFrames."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for i, R in enumerate(matrices):
            n = R.shape[0]
            pd.DataFrame(R, index=[f'After_Task_{k + 1}' for k in range(n)],
                         columns=[f'Task_{j + 1}' for j in range(n)]).to_excel(writer, sheet_name=f'Run_{i + 1}')
        for name, df in (extra_sheets or {}).items():
            df.to_excel(writer, sheet_name=name, index=False)
