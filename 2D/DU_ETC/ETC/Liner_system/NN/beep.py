
# beep.py
import sys
import time

def beep():
    if sys.platform.startswith('win'):
        import winsound
        winsound.Beep(1000, 500)
        winsound.Beep(1000, 500)
        winsound.Beep(1000, 500)
    else:
        print('\a')  # ターミナルベル
        time.sleep(0.5)

if __name__ == "__main__":
    beep()
