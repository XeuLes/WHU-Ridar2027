#!/usr/bin/env bash

set +u

PROJECT_DIR="/home/user/code/T-DT_Radar-main"
LOG_DIR="${PROJECT_DIR}/logs"
SCREEN_NAME="radar_autostart"

# 等桌面自启动项加载稳定
sleep 8

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}" || exit 1

# 清理死亡的 screen socket
screen -wipe >/dev/null 2>&1 || true

# 如果已经有同名 screen，就不重复启动
if screen -list | grep -qE "[0-9]+\.${SCREEN_NAME}[[:space:]]"; then
    echo "[INFO] screen ${SCREEN_NAME} already exists, skip."
    exit 0
fi

LOG_FILE="${LOG_DIR}/$(date "+%Y-%m-%d_%H-%M-%S").screenlog"

screen \
    -dmS "${SCREEN_NAME}" \
    -L \
    -Logfile "${LOG_FILE}" \
    bash -lc "
        cd '${PROJECT_DIR}' || exit 1
        exec '${PROJECT_DIR}/watchdog_radar.sh'
    "

echo "[INFO] started screen ${SCREEN_NAME}"
echo "[INFO] log file: ${LOG_FILE}"
screen -ls