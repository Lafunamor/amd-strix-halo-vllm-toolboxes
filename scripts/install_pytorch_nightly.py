#!/usr/bin/env python3
import urllib.request
import re
import os
import sys
import subprocess

date_str = os.environ.get('ROCM_NIGHTLY_DATE', '').strip()
base_url = 'https://rocm.nightlies.amd.com/v2-staging/gfx1151'

if not date_str:
    print("ROCM_NIGHTLY_DATE is empty, defaulting to latest PyTorch nightlies.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--index-url", f"{base_url}/", "--pre", "torch", "torchaudio", "torchvision"])
    sys.exit(0)

print(f"Resolving PyTorch wheels for ROCm Nightly date: {date_str}")
pkgs = ['torch', 'torchaudio', 'torchvision']
urls = []

for p in pkgs:
    index_url = f"{base_url}/{p}/"
    print(f"Fetching: {index_url}")
    try:
        html = urllib.request.urlopen(index_url).read().decode('utf-8')
    except Exception as e:
        print(f"FATAL: Failed to fetch {index_url}: {e}", file=sys.stderr)
        sys.exit(1)
        
    pattern = re.compile(fr'href="\.\./({p}[^"]*a{date_str}-cp312-cp312-linux_x86_64\.whl)"')
    match = pattern.search(html)
    
    if not match:
        print(f"FATAL: Could not find {p} wheel matching date '{date_str}' in {index_url}", file=sys.stderr)
        sys.exit(1)
    
    wheel_name = match.group(1)
    final_url = f"{base_url}/{wheel_name}"
    urls.append(final_url)

print("Found wheels:")
for u in urls:
    print(f" - {u}")

print("Installing PyTorch wheels via pip...")
subprocess.check_call([sys.executable, "-m", "pip", "install"] + urls)
