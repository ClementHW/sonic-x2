#!/usr/bin/env bash
set -eu
trap 'sleep 0.2; printf "complete\\n" > "${X2_TEST_HANDOFF_MARKER:?}"; exit 130' INT TERM
printf 'ready\n' > "${X2_TEST_READY_MARKER:?}"
if [[ "${X2_TEST_REQUIRE_CONFIRM:-0}" == 1 ]]; then
    IFS= read -r reply
    printf '%s\n' "${reply}" > "${X2_TEST_INPUT_MARKER:?}"
    [[ "${reply}" == y ]] || exit 3
fi
while true; do
    sleep 0.1
done
