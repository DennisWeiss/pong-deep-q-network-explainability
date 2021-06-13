import gym
from ale_py import ALEInterface

ale = ALEInterface()


#ENVIRONMENT = "Pong2Player"
#ENVIRONMENT = "Pong-v0"
#environment = gym.make(ENVIRONMENT)  # Get env

#roms = 'roms/pong.bin'
#ale = ALEInterface(roms.encode('utf-8'))

#cur_total_points = ale.ale_getPoints()
#cur_paddle_bounce = ale.ale_getSideBouncing()

#print(cur_paddle_bounce)

from gym import envs

print(envs.registry.all())

#environment = gym.make('Pong-v0')  # Get env

#İnstall Atari Learning Environment
# vcpkg install zlib sdl1