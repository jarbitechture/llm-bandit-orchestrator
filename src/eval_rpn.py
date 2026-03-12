import math

_OPS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: math.trunc(a / b),
}


def eval_rpn(tokens: list[str]) -> int:
    if not tokens:
        raise ValueError("empty expression")
    stack: list[int] = []
    for tok in tokens:
        if tok in _OPS:
            if len(stack) < 2:
                raise ValueError(f"insufficient operands for '{tok}'")
            b, a = stack.pop(), stack.pop()
            stack.append(_OPS[tok](a, b))
        else:
            try:
                stack.append(int(tok))
            except ValueError:
                raise ValueError(f"unknown token: {tok}")
    if len(stack) != 1:
        raise ValueError(f"leftover operands: {stack}")
    return stack[0]
