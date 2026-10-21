# 3D PID Control

Python desktop application for 3D PID control, camera based tracking, and device communication.

## Create the Conda environment

```powershell
conda env create -f environment.yml
conda activate 3d-pid
```

If the environment already exists, update it with:

```powershell
conda env update -f environment.yml --prune
```

## Run

From this directory, run:

```powershell
python main.py
```

The Hikrobot side camera requires the separately installed Hikrobot MVS SDK. Place its `MvImport` Python package in this project directory or its parent directory, and install the MVS runtime using Hikrobot's installer. Camera and serial device access also requires the corresponding hardware and drivers.
