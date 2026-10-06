import genesis as gs

from .implementation import main

if __name__ == "__main__":
    try:
        main()
    finally:
        gs.destroy()
