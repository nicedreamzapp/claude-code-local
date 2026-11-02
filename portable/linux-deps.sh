#!/bin/bash
# System libraries the prebuilt llama.cpp needs on Linux. Sourced by portable/setup-linux.sh.
# Stock Ubuntu 24.04 has no libgomp.so.1 (GNU OpenMP), and llama-server won't start without it.

# Is libgomp.so.1 already known to the dynamic linker?
ccl_have_libgomp() {
  local ldc
  for ldc in "$(command -v ldconfig 2>/dev/null)" /sbin/ldconfig /usr/sbin/ldconfig; do
    [ -n "$ldc" ] && [ -x "$ldc" ] || continue
    "$ldc" -p 2>/dev/null | grep -q 'libgomp\.so\.1' && return 0
    return 1
  done
  return 1
}

# Install libgomp1 with apt-get (through sudo when not root). Returns 1 with a clear message
# when it can't: no apt-get (not Debian/Ubuntu), or no root and no sudo.
ccl_ensure_libgomp() {
  ccl_have_libgomp && return 0
  echo "llama.cpp needs libgomp.so.1 (GNU OpenMP), which this system doesn't have."
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "  apt-get isn't available here, so install your distribution's libgomp package yourself:"
    echo "    Fedora/RHEL: sudo dnf install libgomp    Arch: sudo pacman -S gcc-libs"
    echo "    openSUSE: sudo zypper install libgomp1"
    return 1
  fi
  local sudo=""
  if [ "$(id -u)" != 0 ]; then
    if command -v sudo >/dev/null 2>&1; then sudo="sudo"
    else
      echo "  Not root and sudo isn't installed. As root, run:  apt-get install -y libgomp1"
      return 1
    fi
  fi
  echo "  Installing it: ${sudo:+$sudo }apt-get install -y libgomp1"
  if $sudo apt-get install -y libgomp1 || { $sudo apt-get update && $sudo apt-get install -y libgomp1; }; then
    return 0
  fi
  echo "  That didn't work. Run it yourself, then start claude-local:  sudo apt-get install -y libgomp1"
  return 1
}

# Shared libraries a binary needs but can't find (ldd's "not found" lines), space separated.
ccl_missing_libs() {
  command -v ldd >/dev/null 2>&1 || return 0
  ldd "$1" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ' | sed 's/ $//'
}
