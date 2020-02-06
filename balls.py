import cv2
from argparse import ArgumentParser
import time
from math import atan2
import numpy as np

parser = ArgumentParser()
parser.add_argument('--green', default=100)
parser.add_argument('--red', default=260)
parser.add_argument('--white', default=200)
parser.add_argument('--width', default=100)
parser.add_argument('--height', default=100)
parser.add_argument('--file', default=None)
parser.add_argument('--debug', default=True)
parser.add_argument('--k', default=1.025)
args = parser.parse_args()

GREEN_THR = args.green
RED_THR = args.red
WIDTH = args.width
HEIGHT = args.height
FILE = args.file
DEBUG = args.debug
WHITE = args.white
K = args.k

if FILE is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(FILE)

G = 1
R = 0

COL_THRS = [255] * 3
COL_THRS[G] = GREEN_THR
COL_THRS[R] = RED_THR

def calc_angle(xg, yg, xr, yr):
    x_up = x
    y_up = 0
    v1 = (x_up - xg, y_up - yg)
    v2 = (xr - xg, yr - yg)
    return atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(ret, frame)
    xs = [0] * 3
    ys = [0] * 3
    cnts = [0] * 3

    X_step = int(frame.shape[1] / WIDTH + 1)
    Y_step = int(frame.shape[0] / HEIGHT + 1)

    if DEBUG:
        frame2 = np.zeros((HEIGHT + 3, WIDTH + 3, 3))
        #print(frame2.shape)

    for y in range(0, frame.shape[0], Y_step):
        for x in range(0, frame.shape[1], X_step):
            for c in range(3):
                Z =  list(frame[y][x])
                Z.pop(c)
                Z = max(Z)
                if WHITE >= frame[y][x][c] >= COL_THRS[c] and Z * K < frame[y][x][c]:
                    print(frame[y][x], c)
                    print(Z * 2, frame[y][x][c])
                    cnts[c] += 1
                    xs[c] += x
                    ys[c] += y
                    #print(frame.shape[0], y, Y_step)
                    frame2[y // Y_step][x // X_step][c] = frame[y][x][c]
    #xs[R] /= cnts[R]
    xs[G] /= cnts[G]
    #ys[R] /= cnts[R]
    ys[G] /= cnts[G]
    print('---$$$---')
    print(xs[R], ys[R], cnts[R])
    print(xs[G], ys[G], cnts[G])
    print(calc_angle(xs[G], ys[G], xs[R], ys[R]) / 3.14 * 180)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
