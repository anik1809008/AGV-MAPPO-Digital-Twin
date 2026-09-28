from src.safety.m7_controller import M7Controller
from src.environment.simulator import GroundTruthSimulator
from src.marl.multi_agent_buffer import MultiAgentRolloutBuffer
from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.real_training_step import run_real_training_step
def test_run_real_training_step_stores_buffer_data():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 1),
        },
        agent_goals={
            0: (2, 0),
            1: (0, 1),
        },
    )

    actor = ActorNetwork(
        input_dim=249,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=498,
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
    )

    result = run_real_training_step(
        actor=actor,
        critic=critic,
        simulator=simulator,
        method="M3",
        trusted_positions={
            0: (0, 0),
            1: (2, 1),
        },
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        multi_agent_buffer=buffer,
    )

    assert len(buffer) == 1
    assert len(buffer.agent_buffers[0]) == 1
    assert len(buffer.agent_buffers[1]) == 1
    assert len(result["actions"]) == 2
    assert len(result["rewards"]) == 2
def test_m7_updates_progress_monitor_but_m4_does_not():
    from src.safety.m4_controller import M4Controller
    from src.safety.shield import SafetyShield
    class DummyMonitor:
        def __init__(self):
            self.calls = []

        def update(self, agent_id, position, goal):
            self.calls.append(
                (agent_id, position, goal)
            )

        def is_stagnating(self, agent_id):
            return False
        def __init__(self):
            self.calls = []

        def update(self, agent_id, position, goal):
            self.calls.append(
                (agent_id, position, goal)
            )

    grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]

    actor = ActorNetwork(
        input_dim=249,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=498,
    )
    m4_controller = M4Controller(
        actor=actor,
        shield=SafetyShield(grid),
    )

    m7_controller = M7Controller(
         actor=actor,
         shield=SafetyShield(grid),
    )
    simulator_m7 = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 1),
        },
        agent_goals={
            0: (2, 0),
            1: (0, 1),
        },
    )

    monitor = DummyMonitor()

    run_real_training_step(
        actor=actor,
        critic=critic,
        simulator=simulator_m7,
        method="M7",
        trusted_positions={
            0: (0, 0),
            1: (2, 1),
        },
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        m7_controller=m7_controller,
        m7_progress_monitor=monitor,
    )

    assert monitor.calls == [
        (0, (0, 0), (2, 0)),
        (1, (2, 1), (0, 1)),
    ]

    simulator_m4 = GroundTruthSimulator(
        grid=grid,
        agent_positions={
            0: (0, 0),
            1: (2, 1),
        },
        agent_goals={
            0: (2, 0),
            1: (0, 1),
        },
    )

    monitor.calls.clear()

    run_real_training_step(
        actor=actor,
        critic=critic,
        simulator=simulator_m4,
        method="M4",
        trusted_positions={
            0: (0, 0),
            1: (2, 1),
        },
        reachable_occupancies={
            0: {(0, 0)},
            1: {(2, 1)},
        },
        aoi_values={
            0: 0,
            1: 0,
        },
        m4_controller=m4_controller,
        m7_progress_monitor=monitor,
    )

    assert monitor.calls == []
