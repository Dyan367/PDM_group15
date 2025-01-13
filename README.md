# PDM_group15

This is the repository of group 15 for the final assignment of RO47005 Planning & Decision Making. The project entails a motion planner validated in a simulation environment for a quadrotor. The simulation environment can be found here [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones). This repository adds additional aviary classes with path planners built upon the `BaseAviary` class within `gym-pybullet-drones`. The MPC implementation uses [`tinympc`](https://tinympc.org/).

## Installation

### Installation of environment

Tested on Intel x64/Ubuntu 22.04 and Apple Silicon/macOS 14.1.

```sh
mkdir group15
cd group15/

git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

### Installation of Motion Planner

First clone the git repository.

```sh
cd group15/
git https://github.com/Dyan367/PDM_group15.git

```

Install dependencies.

```sh
conda activate drones
pip install tinympc
```

## Use

In the directory `PDM_group15/scripts/` there is multiple scripts used to demonstrate the planner. The baseline planner is run using the following:

```sh
conda activate drones
python PDM_group15/scripts/run_simulation_PID_baseline.py
```

To run the MPC implementation please use the following commands:

```sh
conda activate drones
python PDM_group15/scripts/run_simulation_tinympc_dynamic2.py
```
