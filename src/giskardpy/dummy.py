#!/usr/bin/env python
import sympy as sp

def check_sympy_installation(a_value, b_value):
	a = sp.Symbol('a')
	b = sp.Symbol('b')
	f = a**2 + b
	f_a = sp.diff(f, a)
	return f_a.subs({a:a_value, b:b_value})
