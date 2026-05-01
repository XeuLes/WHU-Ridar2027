#!/usr/bin/env bash

set +u

PROJECT_DIR="/home/user/code/T-DT_Radar-main"
LOG_DIR="${PROJECT_DIR}/logs"
ROS_PACKAGE="radar_bringup"
ROS_LAUNCH="radar_launch.py"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}" || {
    echo "[ERROR] Cannot cd to ${PROJECT_DIR}"
    exit 1
}

# ==============================
# 图形环境
# ==============================

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export QT_QPA_PLATFORM=xcb

# 不写死 DISPLAY，优先用当前桌面继承的 DISPLAY；
# 如果没有，则自动探测 :0/:1/:2/:3
if [ -z "${DISPLAY:-}" ] || ! DISPLAY="${DISPLAY}" xset q >/dev/null 2>&1; then
    FOUND_DISPLAY=""

    for d in :0 :1 :2 :3; do
        if DISPLAY="${d}" xset q >/dev/null 2>&1; then
            FOUND_DISPLAY="${d}"
            break
        fi
    done

    if [ -n "${FOUND_DISPLAY}" ]; then
        export DISPLAY="${FOUND_DISPLAY}"
    else
        export DISPLAY="${DISPLAY:-:0}"
    fi
fi

# XAUTHORITY 优先保持桌面继承值；没有的话再兜底
if [ -z "${XAUTHORITY:-}" ]; then
    if [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    elif [ -f "/home/user/.Xauthority" ]; then
        export XAUTHORITY="/home/user/.Xauthority"
    fi
fi

echo "========================================"
echo "[WATCHDOG] start at $(date)"
echo "[USER] $(whoami)"
echo "[PROJECT_DIR] ${PROJECT_DIR}"
echo "[DISPLAY] ${DISPLAY}"
echo "[XAUTHORITY] ${XAUTHORITY:-}"
echo "[XDG_SESSION_TYPE] ${XDG_SESSION_TYPE:-}"
echo "[XDG_RUNTIME_DIR] ${XDG_RUNTIME_DIR}"
echo "[DBUS_SESSION_BUS_ADDRESS] ${DBUS_SESSION_BUS_ADDRESS}"
echo "========================================"

echo "[INFO] wait 10 seconds for desktop/devices..."
sleep 10

RESTART_COUNT=0

while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))

    echo "========================================"
    echo "[WATCHDOG] launching ROS2 at $(date)"
    echo "[WATCHDOG] restart count: ${RESTART_COUNT}"
    echo "[COMMAND] zsh env + ros2 launch ${ROS_PACKAGE} ${ROS_LAUNCH}"
    echo "========================================"

    zsh -lc "
        set +u

        # 关键：加载你平时手动终端里的 zsh 环境
        if [ -f /home/user/.zshrc ]; then
            source /home/user/.zshrc >/dev/null 2>&1 || true
        fi

        cd '${PROJECT_DIR}' || exit 1

        # 重新继承 watchdog 已经探测好的图形环境
        export DISPLAY='${DISPLAY}'
        export XAUTHORITY='${XAUTHORITY:-}'
        export XDG_RUNTIME_DIR='${XDG_RUNTIME_DIR}'
        export DBUS_SESSION_BUS_ADDRESS='${DBUS_SESSION_BUS_ADDRESS}'
        export QT_QPA_PLATFORM=xcb

        # 使用 zsh 对应的 ROS2 环境
        if [ -f /opt/ros/humble/setup.zsh ]; then
            source /opt/ros/humble/setup.zsh
        else
            source /opt/ros/humble/setup.bash
        fi

        if [ -f '${PROJECT_DIR}/install/setup.zsh' ]; then
            source '${PROJECT_DIR}/install/setup.zsh'
        elif [ -f '${PROJECT_DIR}/install/setup.bash' ]; then
            source '${PROJECT_DIR}/install/setup.bash'
        else
            echo '[ERROR] install setup file not found'
            exit 1
        fi

        echo '[ZSH ENV] DISPLAY=' \$DISPLAY
        echo '[ZSH ENV] XAUTHORITY=' \$XAUTHORITY
        echo '[ZSH ENV] LD_LIBRARY_PATH=' \$LD_LIBRARY_PATH
        echo '[ZSH ENV] PATH=' \$PATH

        echo '[CHECK] dynamic libraries for pnp_calib_node_new:'
        ldd '${PROJECT_DIR}/install/camera_field_calib_tool/lib/camera_field_calib_tool/pnp_calib_node_new' | grep 'not found' || true

        exec ros2 launch '${ROS_PACKAGE}' '${ROS_LAUNCH}'
    "

    EXIT_CODE=$?

    echo "========================================"
    echo "[WATCHDOG] ros2 launch exited at $(date)"
    echo "[WATCHDOG] exit code: ${EXIT_CODE}"
    echo "[WATCHDOG] restart after 10 seconds..."
    echo "========================================"

    sleep 10
done