#!/usr/bin/env bash
# Downloads three@0.180.0 (build/ + examples/jsm/) and the WebXR controller
# glTF models (@webxr-input-profiles/assets, dist/profiles/ only) into
# viewer/vendor/. Idempotent: skips each download if its VERSION file
# already matches.
set -euo pipefail

THREE_VERSION="0.180.0"
PROFILES_VERSION="1.0.13"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIEWER_DIR="${SCRIPT_DIR}/../viewer"
VENDOR_DIR="${VIEWER_DIR}/vendor/three"
VERSION_FILE="${VENDOR_DIR}/VERSION"
PROFILES_DIR="${VIEWER_DIR}/vendor/webxr-input-profiles"
PROFILES_VERSION_FILE="${PROFILES_DIR}/VERSION"

fetch_pack() {
  # fetch_pack <package@version> <dest_tmp_dir> -> prints path to extracted package/ dir
  local spec="$1" dest="$2"
  if command -v npm &>/dev/null; then
    (cd "${dest}" && npm pack "${spec}" --silent >/dev/null)
    tar -xzf "${dest}"/*.tgz -C "${dest}"
  else
    local name="${spec%@*}" version="${spec##*@}"
    curl -fsSL "https://registry.npmjs.org/${name}/-/$(basename "${name}")-${version}.tgz" -o "${dest}/pkg.tgz"
    tar -xzf "${dest}/pkg.tgz" -C "${dest}"
  fi
  echo "${dest}/package"
}

if [[ -f "${VERSION_FILE}" ]] && [[ "$(cat "${VERSION_FILE}")" == "${THREE_VERSION}" ]]; then
  echo "vendor-three: three@${THREE_VERSION} already vendored at ${VENDOR_DIR}, skipping."
else
  TMP_DIR="$(mktemp -d)"
  echo "vendor-three: fetching three@${THREE_VERSION}..."
  PKG_DIR="$(fetch_pack "three@${THREE_VERSION}" "${TMP_DIR}")"

  rm -rf "${VENDOR_DIR}"
  mkdir -p "${VENDOR_DIR}"
  cp -r "${PKG_DIR}/build" "${VENDOR_DIR}/build"
  mkdir -p "${VENDOR_DIR}/examples"
  cp -r "${PKG_DIR}/examples/jsm" "${VENDOR_DIR}/examples/jsm"
  echo "${THREE_VERSION}" > "${VERSION_FILE}"
  rm -rf "${TMP_DIR}"

  echo "vendor-three: done. Vendored to ${VENDOR_DIR}"
  du -sh "${VENDOR_DIR}"
fi

if [[ -f "${PROFILES_VERSION_FILE}" ]] && [[ "$(cat "${PROFILES_VERSION_FILE}")" == "${PROFILES_VERSION}" ]]; then
  echo "vendor-three: @webxr-input-profiles/assets@${PROFILES_VERSION} already vendored at ${PROFILES_DIR}, skipping."
else
  TMP_DIR2="$(mktemp -d)"
  echo "vendor-three: fetching @webxr-input-profiles/assets@${PROFILES_VERSION}..."
  PKG_DIR2="$(fetch_pack "@webxr-input-profiles/assets@${PROFILES_VERSION}" "${TMP_DIR2}")"

  rm -rf "${PROFILES_DIR}"
  mkdir -p "${PROFILES_DIR}"
  # Only dist/profiles/ (controller glTF models + profilesList.json) —
  # XRControllerModelFactory.setPath() points at this dir. Source/build
  # tooling in the npm package isn't needed at runtime.
  cp -r "${PKG_DIR2}/dist/profiles" "${PROFILES_DIR}/profiles"
  echo "${PROFILES_VERSION}" > "${PROFILES_VERSION_FILE}"
  rm -rf "${TMP_DIR2}"

  echo "vendor-three: done. Vendored to ${PROFILES_DIR}"
  du -sh "${PROFILES_DIR}"
fi
