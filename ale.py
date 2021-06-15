from xitari_python_interface import ALEInterface

roms = 'roms/Pong.bin'

ale = ALEInterface(roms.encode('utf-8'))

ale.ale_getSideBouncing()

#cur_total_points = ale.ale_getPoints()
#cur_paddle_bounce = ale.