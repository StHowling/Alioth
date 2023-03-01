import subprocess
import sys


if __name__ == "__main__":
    date = sys.argv[1]

    cmd = ['date -s "'+date+'"'] # date expected to be "yy-mm-dd hh:mm:ss" string
    p=subprocess.Popen(cmd,shell=True)
    p.wait()

