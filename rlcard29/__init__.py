"""
File: rlcard29/__init__.py
Author: Arnob Das
Date: 2025-06-28
"""
    
from rlcard.envs.registration import register
from .envs.twenty_nine import TwentyNineEnv

register(
    env_id='twenty_nine',
    entry_point='rlcard29.envs.twenty_nine:TwentyNineEnv',
) 