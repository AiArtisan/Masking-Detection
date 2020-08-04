import os
import shutil
from distutils.dir_util import copy_tree

# global variables
board = os.environ['BOARD']

# check whether board is supported
def check_env():
    if not board == 'Ultra96':
        raise ValueError("Board {} is not supported.".format(board))
        
check_env()
os.system('chmod 777 1.compile.sh')
compile = os.system('./1.compile.sh')
print(compile)

'''
    name="mask detection",
    version='1.0',
    install_requires='pynq>=2.5 and support DPU',
    url='https://github.com/seujingwei/Masking-Detection',
    author="Jingwei Zhang",
    author_email="993987093@qq.com"
'''