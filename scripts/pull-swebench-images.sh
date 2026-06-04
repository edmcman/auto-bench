#!/usr/bin/env bash
# Pull all SWE-bench eval images referenced in cached harbor task packages.
# Skips images already present locally. Pulls serially to avoid Docker Hub rate limits.
set -euo pipefail

HARBOR_CACHE="${HARBOR_CACHE:-/home/vscode/.cache/harbor/tasks/packages/swe-bench}"
PARALLEL="${PARALLEL:-1}"

images=$(find "$HARBOR_CACHE" -name "Dockerfile" \
    | xargs grep -h "^FROM swebench/" \
    | sed 's/^FROM //' \
    | sort -u)

total=$(echo "$images" | wc -l)
echo "Found $total unique SWE-bench images."

to_pull=()
for img in $images; do
    if docker image inspect "$img" &>/dev/null; then
        echo "  [cached] $img"
    else
        to_pull+=("$img")
    fi
done

echo "${#to_pull[@]} images need pulling."
if [[ ${#to_pull[@]} -eq 0 ]]; then
    echo "All images already present."
    exit 0
fi

pulled=0
failed=0
for img in "${to_pull[@]}"; do
    echo "[$((pulled + failed + 1))/${#to_pull[@]}] Pulling $img ..."
    if docker pull "$img"; then
        ((pulled++)) || true
    else
        echo "  FAILED: $img" >&2
        ((failed++)) || true
    fi
done

echo ""
echo "Done. Pulled: $pulled, Failed: $failed"
[[ $failed -eq 0 ]]
