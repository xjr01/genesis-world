set -eu
project=/data1/xjr/genesis-world-demos
archive="$project/backups/code"
mkdir -p "$archive"
exec 9>"$project/backups/backup.lock"
flock -n 9 || exit 0
stamp=$(date -u +%Y%m%dT%H%M%S%NZ)
staging="$archive/.building-$stamp"
snapshot="$archive/$stamp"
mkdir "$staging"
set -- -a --exclude=/out --exclude=/outputs --exclude='/*.mp4' \
    --exclude=.git --exclude=.agents --exclude=.codex --exclude=.remote-sync \
    --exclude=.aws --exclude=.ssh --exclude=.env --exclude=.venv --exclude=venv \
    --exclude=__pycache__ --exclude=.cache --exclude=.pytest_cache --exclude=.ruff_cache \
    --exclude=.mypy_cache --exclude=build --exclude=dist --exclude=tmp \
    --exclude='*.egg-info' --exclude='*.pyc' --exclude='*.pyd' --exclude='*.so' --exclude='*.dll'
if test -L "$archive/latest"; then
    previous=$(readlink -f "$archive/latest")
    case "$previous" in "$archive"/*) set -- "$@" "--link-dest=$previous" ;; *) exit 1 ;; esac
fi
rsync "$@" "$project/workspace/" "$staging/"
mv "$staging" "$snapshot"
ln -s "$stamp" "$archive/.latest-$stamp"
mv -Tf "$archive/.latest-$stamp" "$archive/latest"
find "$archive" -mindepth 1 -maxdepth 1 -type d -name '20*Z' -mmin +20160 -print0 |
    while IFS= read -r -d '' expired; do
        case "$(realpath "$expired")" in "$archive"/20*Z) rm -rf -- "$expired" ;; *) exit 1 ;; esac
    done
printf 'Snapshot: %s\n' "$snapshot"
