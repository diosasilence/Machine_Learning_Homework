import sympy as sp

n = sp.symbols('n', positive=True, integer=True)
expr = (1 + 1 / n) ** n
result = sp.limit(expr, n, sp.oo)

print(result)
print(float(result))