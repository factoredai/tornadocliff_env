from gymnasium.envs.registration import register

from .tornadocliffenv import TornadoCliffEnv  # noqa :


register(
    id="factoredai/TornadoCliff-v0",
    entry_point="tornadocliff_env.tornadocliffenv:TornadoCliffEnv",
)
