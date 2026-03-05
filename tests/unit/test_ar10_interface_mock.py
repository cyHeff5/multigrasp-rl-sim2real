from src.real.ar10_interface import AR10Interface


def test_ar10_interface_mock():
    interface = AR10Interface()
    assert isinstance(interface.read_q_measured(), list)
