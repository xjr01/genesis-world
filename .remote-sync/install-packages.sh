set -eu
project=/data1/xjr/genesis-world-demos
source "$project/tools/remote-env.sh"
export UV_CACHE_DIR="$project/cache/uv"
export UV_CONCURRENT_DOWNLOADS=16
uv=/home/user/.local/bin/uv
"$uv" pip install --python "$project/envs/genesis-world/bin/python" \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --torch-backend cu126 --requirement "$project/tools/requirements-linux.txt"
"$uv" pip install --python "$project/envs/genesis-world/bin/python" --no-build-isolation \
    --index-url https://mirrors.aliyun.com/pypi/simple/ --torch-backend cu126 \
    --constraint "$project/tools/requirements-linux.txt" --editable "$project/workspace"
"$uv" pip check --python "$project/envs/genesis-world/bin/python"
printf '%s\n' 'ENVIRONMENT_READY'
