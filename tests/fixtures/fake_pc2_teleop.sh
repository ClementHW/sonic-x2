#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == stop ]]; then
    test -f "${X2_TEST_HANDOFF_MARKER:?}"
    printf 'stopped\n' > "${X2_TEST_STOP_MARKER:?}"
fi
