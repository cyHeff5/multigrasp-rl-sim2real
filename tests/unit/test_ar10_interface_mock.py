from src.real.ar10_interface import AR10Interface


def test_read_q_measured_returns_list():
    iface = AR10Interface()
    assert isinstance(iface.read_q_measured(), list)


def test_mock_read_returns_last_target():
    iface = AR10Interface()
    target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iface.send_q_target(target)
    assert iface.read_q_measured() == target


def test_mock_error_is_zero_after_send():
    # In mock mode read_q_measured returns q_target → error must be 0.
    iface = AR10Interface()
    iface.send_q_target([0.5] * 10)
    assert iface.position_error_norm() == 0.0


def test_error_norm_bounded():
    iface = AR10Interface()
    iface.send_q_target([1.0] * 10)
    err = iface.position_error_norm()
    assert 0.0 <= err <= 1.0


def test_send_q_target_clamps_values():
    iface = AR10Interface()
    iface.send_q_target([-0.5] * 5 + [1.5] * 5)
    q = iface.read_q_measured()
    assert all(0.0 <= v <= 1.0 for v in q)


def test_send_q_target_wrong_length():
    iface = AR10Interface()
    try:
        iface.send_q_target([0.5] * 5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
