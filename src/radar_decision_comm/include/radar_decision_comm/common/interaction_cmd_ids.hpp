#pragma once

#include <cstdint>

namespace radar_decision_comm::referee_cmd_ids
{

static constexpr uint16_t kRobotInteraction = 0x0301;
static constexpr uint16_t kMapRadarData = 0x0305;

}  // namespace radar_decision_comm::referee_cmd_ids

namespace radar_decision_comm::interaction_cmd_ids
{

static constexpr uint16_t kRadarDecisionInfo = 0x0121;
static constexpr uint16_t kSentryAlertAngle = 0x0201;
static constexpr uint16_t kSentryFieldState = 0x0202;
static constexpr uint16_t kHeroAlertFlag = 0x0203;
static constexpr uint16_t kDoubleEffectForward = 0x0204;

static constexpr uint16_t kRadarDecisionReceiver = 0x8080;

}  // namespace radar_decision_comm::interaction_cmd_ids
