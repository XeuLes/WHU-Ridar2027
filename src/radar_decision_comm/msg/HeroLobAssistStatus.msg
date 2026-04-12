std_msgs/Header header

uint8 STATE_DISABLED=0
uint8 STATE_WAIT_TRACK=1
uint8 STATE_ACCUMULATING=2
uint8 STATE_WAIT_STABLE=3
uint8 STATE_THREAT_BLOCKED=4
uint8 STATE_SOLVING=5
uint8 STATE_ACTIVE=6
uint8 STATE_CLEARING=7

uint8 state
uint16 hero_robot_id
uint16 nearest_enemy_robot_id

bool hero_found
bool our_track_fresh
bool pose_stable
bool enemy_nearby_3m
bool solution_valid
bool publish_active

float32 window_avg_displacement_m
float32 nearest_enemy_distance_m
string reason
