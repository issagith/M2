from mathtools.calculator import add, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0

def test_add_edge_cases():
    assert add(0, 0) == 0
    assert add(-5, -3) == -8
    assert add(1.5, 2.5) == 4.0

def test_multiply_edge_cases():
    assert multiply(-2, 3) == -6
    assert multiply(2.5, 4) == 10.0