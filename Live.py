import cv2
import numpy as np
import imutils
import tensorflow as tf
from digit_recognizer import scan_sudoku
import sudoku

model = tf.keras.models.load_model('model/digit-recognizer.h5')


webcam = cv2.VideoCapture(0)


def print_output(input_board, solution_board, cv2_frame):
    """
    Prints solution on CV2 frame
    Inputs:
    input_board: Scanned sudoku board
    solution_board: Solved sudoku board
    cv2_frame: Output frame to print board
    """
    input_board = input_board.astype(str)
    solution_board = solution_board.astype(str)
    for i in range(0, 9):
        for j in range(0, 9):
            if (input_board[i][j]) == (solution_board[i][j]):
                solution_board[i][j] = " "
    transformed_board = solution_board[::-1]

    color = (50, 50, 50)
    for j in range(0, 9):
        for i in range(0, 9):
            text = str(transformed_board[j][i])
            org = (int(round(x3+(increment_width_hori*i)+(increment_width_verti*j))),
                   int(round(y3+(increment_height_hori*i)+(increment_height_verti*j))))
            cv2.putText(cv2_frame, text, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)


def binary_Image(a):

    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blur", blur)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    # cv2.imshow("thresh", thresh)

    return thresh


while True:

    key = cv2.waitKey(1)

    ret, frame = webcam.read()
    # cv2.imshow ("Live", frame)

    cnts = cv2.findContours(binary_Image(
        frame), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:

        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        #  TL has smallest Sum, BR has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # TR has smallest difference, BL has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        # Width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # Height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # Final Dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)

        # Warp transform
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        cv2.imshow('Initial Frame', frame)

        # cv2.imshow('Transformed Capture', warp)

        # cv2.imshow('Transformed Capture', binary_Image(warp))

        lines = cv2.HoughLinesP(binary_Image(
            warp), 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # cv2.imshow('HoughP Transform', warp)

        x3 = rect[3, 0]
        y3 = rect[3, 1]

        x2 = rect[2, 0]
        y2 = rect[2, 1]

        x1 = rect[1, 0]
        y1 = rect[1, 1]

        x = rect[0, 0]
        y = rect[0, 1]

        increment_width_hori = (x2-x3+x1-x)/18
        increment_height_hori = (y2-y3+y1-y)/18

        increment_width_verti = (x-x3+x1-x2)/18
        increment_height_verti = (y-y3+y1-y2)/18

        detected_board = scan_sudoku(warp, model)
        solved_board = None

        if solved_board is not None:
            if original_board == detected_board:
                if sudoku.all_board_non_zero(solved_board):
                    print_output(original_board, solved_board, frame)

        original_board = detected_board

        solved_board = original_board

        if original_board is not None:
            solved_board = solved_board.astype(int)
            original_board = original_board.astype(int)
            sudoku.solve_sudoku(solved_board)

            if sudoku.all_board_non_zero(solved_board):
                print_output(original_board, solved_board, frame)

        cv2.imshow("text", frame)

    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
