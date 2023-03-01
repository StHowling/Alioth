import os
import re
import sys

os.system("kill -s 9 `ps -aux | grep fio | awk '{print $2}'`")