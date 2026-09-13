#!/bin/sh
# Install the repo's git hooks. Hooks live in .git/hooks/, which is NOT version
# controlled - so the real copies are kept in hooks/ and symlinked into place.
# A symlink rather than a copy means editing hooks/pre-push takes effect at
# once, and a fresh clone needs one command rather than remembering the content.
set -e
cd "$(git rev-parse --show-toplevel)"
for h in hooks/*; do
  name=$(basename "$h")
  [ "$name" = "install.sh" ] && continue
  ln -sf "../../hooks/$name" ".git/hooks/$name"
  echo "installed .git/hooks/$name -> hooks/$name"
done
