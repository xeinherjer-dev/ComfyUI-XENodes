def generate_frame_indices(num_images: int, pingpong: bool) -> list[int]:
    """
    Generates a list of frame indices for playback, handling pingpong loops.
    """
    if pingpong and num_images > 2:
        return list(range(num_images)) + list(range(num_images - 2, 0, -1))
    return list(range(num_images))
