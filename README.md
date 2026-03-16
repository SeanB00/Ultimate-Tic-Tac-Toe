# Ultimate-Tic-Tac-Toe

Current layout uses a package split plus dedicated asset folders:

- `uttt/game` for game rules, hashing, LMDB helpers, table maintenance, and inspection
- `uttt/ml` for CNN training and training entry modules
- `uttt/ui` for the Kivy application and API-backed move helper
- `uttt/data_build` for dataset generation scripts
- `data/qtable` for the LMDB Q-table
- `data/processed` for generated NumPy datasets and metadata
- `models` for trained checkpoints and plot outputs
- `artifacts` for generated inspection plots

Windows setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install PyTorch separately so you can choose the right GPU build for your machine.
Use the current Windows command from https://pytorch.org/get-started/locally/ and then verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run examples from project root:

- `python -m uttt.ml.train_cnn`
- `python -m uttt.ml.mixed_cnn`
- `python -m uttt.ml.filtered_cnn`
- `python -m uttt.ui.graphics`
- `python -m uttt.ui.api`
- `python -m uttt.game.inspection`
- `python -m uttt.data_build.build_mixed_npy_v1`
- `python -m uttt.data_build.filter_lmdb_to_numpy`

PyCharm script-path runs also work against the packaged files directly, for example:

- `uttt\ml\train_cnn.py`
- `uttt\ml\mixed_cnn.py`
- `uttt\ui\graphics.py`
- `uttt\ui\api.py`
- `uttt\game\inspection.py`
