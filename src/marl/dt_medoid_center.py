"""Select DT observation centers using only controller-visible information."""


def select_dt_medoid_centers(trusted_positions, possible_current_positions):
    centers = {}

    for agent_id, trusted in trusted_positions.items():
        candidates = sorted(set(possible_current_positions.get(agent_id, ())))

        if not candidates:
            raise ValueError(f"Missing possible positions for AGV {agent_id}")

        def score(candidate):
            total = sum(
                abs(candidate[0] - other[0]) +
                abs(candidate[1] - other[1])
                for other in candidates
            )
            trusted_distance = (
                abs(candidate[0] - trusted[0]) +
                abs(candidate[1] - trusted[1])
            )
            return (total, trusted_distance, candidate)

        centers[agent_id] = min(candidates, key=score)

    return centers
