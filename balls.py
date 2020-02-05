import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--green')
parser.add_argument('--red')
parser.add_argument('--width', default=100)
parser.add_argument('--height', default=100)
args = parser.parse_args()

GREEN_THR = args.green
RED_THR = args.red
WIDTH = args.width
HEIGHT = args.height

cap = cv2.VideoCapture(0)

G = 1
R = 0

COL_THRS = [0] * 3
COL_THRS[G] = GREEN_THR
COL_THRS[R] = RED_THR

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    xs = [0] * 3
    ys = [0] * 3
    cnts = [0] * 3
    
    X_step = frame.shape[1] / WIDTH
    Y_step = frame.shape[0] / HEIGHT
    
    for y in range(0, frame.shape[0], Y_step):
        for x in range(0, frame.shape[1], X_step):
            for c in range(3):
                if frame[x][y][c] >= COL_THRS[c]:
                    cnts[c] += 1
                    xs[c] += x
                    ys[c] += y

    xs[R] /= cnts[R]
    xs[G] /= cnts[G]
    ys[R] /= cnts[R]
    ys[G] /= cnts[G]
    print('---$$$---')
    print(xs[R], ys[R], cnts[R])
    print(xs[G], ys[G], cnts[G])

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
