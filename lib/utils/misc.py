import sys, os
from platform import system

def detect_os():
    os = system().lower()
    if 'darwin' in os:
        return "MAC_OS_X"
    elif 'windows' in os:
        return "WINDOWS"
    elif 'linux' in os:
        with open('/proc/version','r') as f:
            vers = f.read()
            if 'microsoft' in vers.lower():
                return "WSL" # Windows10的Linux子系统
        return "LINUX"
    elif 'bsd' in os:
        return "BSD"
    elif 'cygwin' in os:
        return "CYGWIN"
    else:
        return None