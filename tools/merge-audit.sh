#!/usr/bin/env bash
# After merging upstream, prove nothing was silently dropped.
#
# A merge can lose a subsystem two ways and neither fails the build loudly: the file never arrives,
# or it arrives and no parent module declares it, so it is dead code the compiler never sees. This
# checks both against the set of files upstream added since a base commit.
#
#   ./tools/merge-audit.sh [base] [upstream-ref]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
BASE=${1:-0a14af4a}
UP=${2:-upstream-renamed}

git rev-parse --verify -q "$UP" >/dev/null || {
  echo "no ref '$UP'. Build it first: branch upstream/master, rename the crates onto our layout."
  exit 1
}

python3 - "$BASE" "$UP" <<'PY'
import os, re, subprocess, sys
base, up = sys.argv[1], sys.argv[2]
added = subprocess.run(["git","diff","--name-only","--diff-filter=A",base,up],
                       capture_output=True, text=True).stdout.split()
skip = ("docs/","website/","releases/",".github/","examples/","res/","scripts/")
added = [f for f in added if not f.startswith(skip)]

missing = [f for f in added if not os.path.exists(f)]

# cargo auto-discovers tests/, examples/ and benches/; everything under src/ needs a `mod`.
undeclared = []
for f in added:
    if not f.endswith(".rs") or not os.path.exists(f):        continue
    if "/tests/" in f or "/examples/" in f or "/benches/" in f: continue
    d, base_name = os.path.dirname(f), os.path.basename(f)[:-3]
    if base_name in ("mod","lib","main","build"):              continue
    parents = [os.path.join(d,"mod.rs"), os.path.join(d,"lib.rs"), d.rstrip("/")+".rs"]
    if not any(os.path.exists(p) and re.search(rf"\bmod\s+{re.escape(base_name)}\s*[;{{]",
                                               open(p, errors="ignore").read())
               for p in parents):
        undeclared.append(f)

print(f"upstream added {len(added)} files since {base}")
print(f"  absent from our tree : {len(missing)}")
for f in missing[:40]:    print("     ", f)
print(f"  present but no parent module declares them : {len(undeclared)}")
for f in undeclared[:40]: print("     ", f)
print()
print("OK" if not missing and not undeclared else "INCOMPLETE")
PY

echo
echo "and the other direction, what upstream has never had and we must not lose:"
for d in hanzo-router enso hanzo-quant/src/nvfp4 hanzo-quant/kernels/nvfp4 \
         hanzo-paged-attn/src/rocm hanzo-paged-attn/src/vulkan; do
  [ -e "$d" ] && printf '  %-40s present\n' "$d" || printf '  %-40s GONE\n' "$d"
done
