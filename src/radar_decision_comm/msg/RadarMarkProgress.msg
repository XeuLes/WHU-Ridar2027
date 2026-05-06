std_msgs/Header header
# RM2026 0x020C bitfield.
# Unprefixed fields are opponent robot vulnerable/special flags, copied from bits 0-5.
uint8 hero
uint8 engineer
uint8 infantry_3
uint8 infantry_4
uint8 aerial_6
uint8 sentinel
uint8 ally_hero
uint8 ally_engineer
uint8 ally_infantry_3
uint8 ally_infantry_4
uint8 ally_aerial_6
uint8 ally_sentinel
uint16 raw_bits
