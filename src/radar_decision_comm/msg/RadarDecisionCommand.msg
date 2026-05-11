std_msgs/Header header
# Legacy compatibility: command_type=1 and command_value used to mean radar_cmd only.
uint8 command_type
uint8 command_value
uint8 radar_cmd
uint8 password_cmd
uint8[6] password
string note
