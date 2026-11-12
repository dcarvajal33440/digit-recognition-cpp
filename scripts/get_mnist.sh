#!/usr/bin/env bash
# Download MNIST into data/ and decompress it.
# usage: scripts/get_mnist.sh [dest_dir]        (default: data)
# Override mirrors with: MNIST_MIRRORS="https://a/ https://b/" scripts/get_mnist.sh
set -euo pipefail

dest=${1:-data}
mirrors=${MNIST_MIRRORS:-"https://storage.googleapis.com/cvdf-datasets/mnist/ https://ossci-datasets.s3.amazonaws.com/mnist/"}
files="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
# Expected uncompressed sizes in bytes; a wrong size means a truncated or corrupt download.
expected_size() {
    case $1 in
        train-images-idx3-ubyte) echo 47040016 ;;
        train-labels-idx1-ubyte) echo 60008 ;;
        t10k-images-idx3-ubyte)  echo 7840016 ;;
        t10k-labels-idx1-ubyte)  echo 10008 ;;
    esac
}

mkdir -p "$dest"
for f in $files; do
    want=$(expected_size "$f")
    if [[ -f $dest/$f && $(stat -c %s "$dest/$f") -eq $want ]]; then
        echo "ok      $f (already present)"
        continue
    fi
    done_file=0
    for m in $mirrors; do
        tmp=$(mktemp)
        if curl -fsSL --retry 2 -o "$tmp" "${m%/}/$f.gz" && gzip -t "$tmp" 2>/dev/null; then
            gzip -dc "$tmp" > "$dest/$f"
            rm -f "$tmp"
            if [[ $(stat -c %s "$dest/$f") -eq $want ]]; then
                echo "fetched $f"
                done_file=1
                break
            fi
            echo "size mismatch for $f from $m" >&2
            rm -f "$dest/$f"
        else
            rm -f "$tmp"
            echo "failed  $f from $m" >&2
        fi
    done
    if [[ $done_file -eq 0 ]]; then
        echo "error: could not obtain $f from any mirror" >&2
        exit 1
    fi
done
echo "MNIST ready in $dest/"
