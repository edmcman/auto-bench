#!/usr/bin/env bash
set -euo pipefail

TARGET_CONFIG_FILE="$HOME/.docker/config.json"

mkdir -p "$(dirname "$TARGET_CONFIG_FILE")"
cat > "$TARGET_CONFIG_FILE" <<'EOF'
{
  "proxies": {
    "default": {
      "httpProxy": "http://cloudproxy.sei.cmu.edu:80",
      "httpsProxy": "http://cloudproxy.sei.cmu.edu:80",
      "ftpProxy": "http://cloudproxy.sei.cmu.edu:80",
      "noProxy": ".sei.cmu.edu,.cert.org,172.16.0.0/16,10.64.0.0/16,10.60.0.0/16,localhost,127.0.0.1,172.17.0.0/16,172.17.0.1,172.17.0.2"
    }
  }
}
EOF
chmod 600 "$TARGET_CONFIG_FILE"
