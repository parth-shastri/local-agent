"""
User-defined python tools that model can use, (Python functions)
"""
from src.tools.functions.stock_analysis_functions import (
    analyse_company_yf
)

# define the tools


def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result integer.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b
    """
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The quotient of a divided by b.
    """
    return a // b


__all__ = ["multiply", "add", "subtract", "divide", "analyse_company_yf"]
