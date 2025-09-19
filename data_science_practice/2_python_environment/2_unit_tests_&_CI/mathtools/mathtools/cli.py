import sys
from .calculator import add, multiply

def main():
    if len(sys.argv) != 4:
        print("Usage: mathcalc <operation> <num1> <num2>")
        print("Operations: add, multiply")
        return

    op, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

    if op == "add":
        result = add(a, b)
    elif op == "multiply":
        result = multiply(a, b)
    else:
        print("Unknown operation")
        return

    print(f"{a} {op} {b} = {result}")