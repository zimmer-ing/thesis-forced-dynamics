# Environment setup

[Project overview](../README.md)

Use Mamba and the lock file for your operating system and architecture.
The locks specify the Python packages and their builds. The YAML files provide
editable dependency lists.

## Install Mamba

[Miniforge3](https://github.com/conda-forge/miniforge) includes Mamba and uses
conda-forge as its only default package channel. Mambaforge has been retired.
Use Miniforge3 for a new installation.

| Platform | Installer |
| --- | --- |
| Linux x86-64 | [Miniforge3-Linux-x86_64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) |
| Apple Silicon Mac | [Miniforge3-MacOSX-arm64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) |

Run the matching installer from the download directory:

```sh
bash Miniforge3-Linux-x86_64.sh   # Linux
bash Miniforge3-MacOSX-arm64.sh  # Apple Silicon Mac
```

For the default installation directory `~/miniforge3`, initialise Mamba for
your shell:

```sh
~/miniforge3/bin/mamba shell init -s bash -r ~/miniforge3  # Linux with Bash
~/miniforge3/bin/mamba shell init -s zsh -r ~/miniforge3   # Mac with Zsh
```

Choose the command for your platform, then reopen the terminal. If Mamba is
already installed, `mamba info` shows its installation and environment paths.
Use that installation directory for shell initialisation.

## Package sources

Both explicit locks contain only conda-forge package URLs. The YAML files use
`conda-forge` and `nodefaults`. The commands below also override configured
channels. Anaconda states that its commercial payment requirements do not apply
to conda-forge. Individual package licences still apply.
See [Anaconda's licensing FAQ](https://www.anaconda.com/legal).

## Linux with NVIDIA GPU

Requires Linux x86-64, glibc 2.28 or newer, and an installed NVIDIA driver
compatible with CUDA 12.6.

```sh
mamba create --name thesis-gpu --file environments/locks/thesis-linux-64-cuda.explicit.txt \
  --override-channels --channel conda-forge
mamba activate thesis-gpu
```

Run these commands from the repository root. CUDA runtime packages are included
in the environment. The NVIDIA driver is installed on the host.

## Apple Silicon Mac

This environment uses CPU PyTorch:

```sh
mamba create --name thesis-cpu --file environments/locks/thesis-macos-arm64.explicit.txt \
  --override-channels --channel conda-forge
mamba activate thesis-cpu
```

The CUDA inference-time measurements and GPU solver comparison require the
Linux GPU environment. Saved measurement files can be evaluated on the Mac.

## Environment files

| Platform | Package lock | Editable dependencies |
| --- | --- | --- |
| Linux x86-64, CUDA | [Linux lock](locks/thesis-linux-64-cuda.explicit.txt) | [Linux YAML](environment.linux-64-cuda.yml) |
| Apple Silicon, CPU | [Mac lock](locks/thesis-macos-arm64.explicit.txt) | [Mac YAML](environment.macos-arm64.yml) |

To create an environment from the editable dependencies instead:

```sh
mamba create --file environments/environment.linux-64-cuda.yml \
  --override-channels --channel conda-forge
mamba activate tii-thesis-linux-64-cuda
```

For Mac, use `environment.macos-arm64.yml` and activate
`tii-thesis-macos-arm64`. A YAML installation may resolve different
transitive package builds from those in the lock. The supplied files are
specific to the two platforms listed above.

## Additional tools

- **Industrial Robot data:** RAR extraction needs `bsdtar`, `unar`, `unrar`
  or `7z`. macOS supplies `bsdtar`.
- **Fractional data regeneration:** MATLAB and FOMCON. The supplied trajectories
  can be used without them.
- **Forced-solver figure:** a LaTeX installation. The research-question figures
  use Matplotlib without external LaTeX.

Continue with [data preparation](../Scripts/DataPreparation/README.md), then
[training](../Scripts/Training/README.md).
