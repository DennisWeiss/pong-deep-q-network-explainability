from ale_py import ALEInterface

ale = ALEInterface()

roms = 'roms/pong.bin'

ale.loadROM(roms)

#cur_total_points = ale.ale_getPoints()
#cur_paddle_bounce = ale.