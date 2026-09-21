import pkg_resources
from subprocess import call

def check_outdated_packages():
    call("pip list --outdated", shell=True)

if __name__ == "__main__":
    check_outdated_packages()
