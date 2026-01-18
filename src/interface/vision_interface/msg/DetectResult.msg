std_msgs/Header header
float32[7] blue_x
float32[7] blue_y
float32[7] red_x
float32[7] red_y

# UAV light bar targets (vertical pairs) - up to 2 targets per UAV
uint8 blue_uav_light_count     # number of blue team light bar pairs (0-2)
float32[2] blue_uav_light_x    # x coordinates of blue light bar pair centers
float32[2] blue_uav_light_y    # y coordinates of blue light bar pair centers

uint8 red_uav_light_count      # number of red team light bar pairs (0-2)
float32[2] red_uav_light_x     # x coordinates of red light bar pair centers
float32[2] red_uav_light_y     # y coordinates of red light bar pair centers