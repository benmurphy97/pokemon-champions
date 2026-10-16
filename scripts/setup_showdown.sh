#!/bin/sh
# Clone and build the Showdown engine into vendor/ (gitignored). Safe to re-run.
# Pinned so results are reproducible; bump the commit when a new regulation needs newer data.
set -e
cd "$(dirname "$0")/.."

COMMIT=3a766f2   # 2026-10-13

if [ ! -d vendor/pokemon-showdown/.git ]; then
	mkdir -p vendor
	git clone https://github.com/smogon/pokemon-showdown.git vendor/pokemon-showdown
fi
cd vendor/pokemon-showdown
git fetch --quiet origin
git checkout --quiet "$COMMIT"
npm install --no-audit --no-fund
node build
echo "Showdown ready at vendor/pokemon-showdown ($COMMIT)"
