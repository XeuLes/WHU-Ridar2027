builtin_interfaces/Time stamp

string state
bool has_target

# measured yaw/pitch in angle_unit
float32 yaw
float32 pitch
string angle_unit

# predicted (latency compensated) yaw/pitch in same unit
float32 yaw_pred
float32 pitch_pred

float32 distance
float32 confidence

bool has_pose
float32 tx
float32 ty
float32 tz

int64 latency_ns

# optional extra latency configured by user (e.g., actuator delay)
int64 extra_latency_ns

bool roi_used
int32 roi_x
int32 roi_y
int32 roi_w
int32 roi_h

uint64 pushed_frames
uint64 processed_frames
uint64 dropped_frames
