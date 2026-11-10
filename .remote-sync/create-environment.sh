set -eu
project=/data1/xjr/genesis-world-demos
export CONDARC="$project/tools/condarc"
export CONDA_ENVS_PATH="$project/envs"
export CONDA_PKGS_DIRS="$project/cache/conda"
export XDG_CACHE_HOME="$project/cache"
export TMPDIR="$project/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR"
conda=/home/user/yizao/rigid060_001/miniconda3/bin/conda
"$conda" create --yes --no-default-packages --override-channels --channel conda-forge \
    --prefix "$project/envs/genesis-world" python=3.11.15 pip
bash "$project/tools/install-packages.sh"
