from .tornadocliffenv import TornadoCliffEnv  # noqa :
from gymnasium.envs.registration import register

register(
    id="factoredai/TornadoCliff-v0",
    entry_point="tornadocliff_env.tornadocliffenv:TornadoCliffEnv",
)