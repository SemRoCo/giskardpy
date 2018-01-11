# USE_SYMENGINE = False
USE_SYMENGINE = True

# BACKEND = None
# BACKEND = 'cython'
BACKEND = 'llvm'
# BACKEND = 'numpy'

def print_wrapper(msg):
	print(msg)