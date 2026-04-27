std_msgs/Header header
float32[7] blue_x
float32[7] blue_y
float32[7] red_x
float32[7] red_y

# Ground robot YOLO bbox in original image coordinates.
# Arrays use the same index as blue_x/red_x: index 0..5 => robot 1..6.
# A box is valid when x2 > x1 and y2 > y1.
float32[7] blue_car_box_x1
float32[7] blue_car_box_y1
float32[7] blue_car_box_x2
float32[7] blue_car_box_y2
float32[7] red_car_box_x1
float32[7] red_car_box_y1
float32[7] red_car_box_x2
float32[7] red_car_box_y2

# UAV YOLO bbox in original image coordinates (valid when w/h > 0)
float32 blue_uav_bbox_x
float32 blue_uav_bbox_y
float32 blue_uav_bbox_w
float32 blue_uav_bbox_h
float32 blue_uav_conf

float32 red_uav_bbox_x
float32 red_uav_bbox_y
float32 red_uav_bbox_w
float32 red_uav_bbox_h
float32 red_uav_conf

# UAV light bar targets (vertical pairs) - up to 2 targets per UAV
uint8 blue_uav_light_count     # number of blue team light bar pairs (0-2)
float32[2] blue_uav_light_x    # x coordinates of blue light bar pair centers
float32[2] blue_uav_light_y    # y coordinates of blue light bar pair centers

uint8 red_uav_light_count      # number of red team light bar pairs (0-2)
float32[2] red_uav_light_x     # x coordinates of red light bar pair centers
float32[2] red_uav_light_y     # y coordinates of red light bar pair centers
