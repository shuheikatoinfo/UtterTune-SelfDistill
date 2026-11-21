# Sourced by the shell scripts: where they find things outside this repository.
# REPO_ROOT is this checkout, found from this file's location. Every other location is an environment
# variable (the list and meanings are in scripts/paths.py). A script names the ones it uses with
#   require_env DATA IRODORI_DIR ...
# before it starts any work, so a missing variable stops it at once rather than hours into a run.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

require_env() {
  local name
  for name in "$@"; do
    [ -n "${!name:-}" ] || { echo "environment variable $name is not set (see scripts/paths.py)" >&2; exit 1; }
  done
}
