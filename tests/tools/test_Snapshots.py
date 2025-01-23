import pytest

from superfv.tools.snapshots import Snapshots


def test_log_and_retrieve():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")

    assert snapshots(1.0) == "data1"
    assert snapshots(2.0) == "data2"


def test_log_duplicate_time():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")

    with pytest.raises(ValueError, match="Snapshot data already exists for time 1.0"):
        snapshots.log(1.0, "data2")


def test_unlog():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.unlog(1.0)

    with pytest.raises(KeyError, match="No snapshot data available for time 1.0"):
        snapshots(1.0)


def test_unlog_nonexistent_time():
    snapshots = Snapshots()
    with pytest.raises(KeyError, match="No snapshot data available for time 1.0"):
        snapshots.unlog(1.0)


def test_clear():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")
    snapshots.clear()

    assert snapshots.size == 0
    assert snapshots.times() == []


def test_getitem():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")

    assert snapshots[0] == "data1"
    assert snapshots[1] == "data2"


def test_getitem_negative_index():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")

    assert snapshots[-1] == "data2"
    assert snapshots[-2] == "data1"


def test_getitem_out_of_range():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")

    with pytest.raises(IndexError, match="Index 1 out of range"):
        snapshots[1]

    with pytest.raises(IndexError, match="Index -2 out of range"):
        snapshots[-2]


def test_iteration():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")

    times = []
    data = []

    for t, d in snapshots:
        times.append(t)
        data.append(d)

    assert times == [1.0, 2.0]
    assert data == ["data1", "data2"]


def test_times():
    snapshots = Snapshots()
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")

    assert snapshots.times() == [1.0, 2.0]


def test_empty_snapshots():
    snapshots = Snapshots()

    assert snapshots.size == 0
    assert snapshots.times() == []

    with pytest.raises(KeyError, match="No snapshot data available for time 1.0"):
        snapshots(1.0)

    with pytest.raises(IndexError, match="Index 0 out of range"):
        snapshots[0]


def test_log_out_of_order():
    snapshots = Snapshots()

    # Logging in reverse order
    snapshots.log(2.0, "data2")
    snapshots.log(1.0, "data1")

    # Ensure the data is correctly stored and accessible even if the log order is reversed
    assert snapshots(1.0) == "data1"
    assert snapshots(2.0) == "data2"

    # Ensure that iteration over snapshots is in chronological order
    times = [t for t, _ in snapshots]
    assert times == [1.0, 2.0]


def test_log_out_of_order_and_access():
    snapshots = Snapshots()

    # Logging in random order
    snapshots.log(3.0, "data3")
    snapshots.log(1.0, "data1")
    snapshots.log(2.0, "data2")

    # Ensure the data is correctly stored and accessible
    assert snapshots(1.0) == "data1"
    assert snapshots(2.0) == "data2"
    assert snapshots(3.0) == "data3"

    # Ensure that iteration over snapshots is in chronological order
    times = [t for t, _ in snapshots]
    assert times == [1.0, 2.0, 3.0]


def test_log_out_of_order_then_remove():
    snapshots = Snapshots()

    # Logging in random order
    snapshots.log(5.0, "data5")
    snapshots.log(1.0, "data1")
    snapshots.log(3.0, "data3")

    # Ensure the data is correctly stored
    assert snapshots(1.0) == "data1"
    assert snapshots(3.0) == "data3"
    assert snapshots(5.0) == "data5"

    # Remove a snapshot and check the order of remaining ones
    snapshots.unlog(3.0)
    assert snapshots(1.0) == "data1"
    assert snapshots(5.0) == "data5"

    # Ensure that iteration reflects the updated data
    times = [t for t, _ in snapshots]
    assert times == [1.0, 5.0]
