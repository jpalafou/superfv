from .device_management import ArrayLike


def check_buffer_slots(buffer: ArrayLike, required: int) -> None:
    """
    Check if the buffer has the required number of slots along the fifth axis.

    Args:
        buffer (ArrayLike): The buffer array to check.
        required (int): The required number of slots.

    Raises:
        ValueError: If the buffer does not have the required number of slots.
    """
    if buffer.shape[4] < required:
        raise ValueError(
            f"Buffer expected to have at least {required} slots along fifth axis, "
            f"got {buffer.shape[4]}"
        )
