"""Select the same recorded frames for surface reconstruction and Isaac capture."""


def replay_indices(frame_count, preview):
    if frame_count < 1:
        raise ValueError("A replay requires at least one recorded frame.")
    if preview:
        return sorted({0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1})
    return list(range(frame_count))
