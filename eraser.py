"""
Interactive image eraser tool.

Usage: python eraser.py --image path/to/your/image.png

    Left click and drag to erase (paints black).
    Press 'a' to increase the brush size.
    Press 'b' to decrease the brush size.
    Press 's' to save changes and exit.
    Press 'x' to exit without saving.
"""
import cv2
import numpy as np
import argparse

# Global variables
drawing = False
brush_size = 20
img = None
clone = None

def erase(event, x, y, flags, param):
    global drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), brush_size, (0, 0, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), brush_size, (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), brush_size, (0, 0, 0), -1)

def main():
    global brush_size, img, clone

    parser = argparse.ArgumentParser(description="Interactive image eraser")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(parser.parse_args())

    image_path = args["image"]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at '{image_path}'")
        return
    
    clone = img.copy()
    
    cv2.namedWindow("Image Eraser")
    cv2.setMouseCallback("Image Eraser", erase)

    print("Press 'a' to increase brush, 'b' to decrease.")
    print("Press 's' to save and exit, 'x' to exit without saving.")

    while True:
        cv2.imshow("Image Eraser", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            brush_size += 5
            print(f"Brush size: {brush_size}")

        elif key == ord("b"):
            brush_size = max(5, brush_size - 5)
            print(f"Brush size: {brush_size}")

        elif key == ord("s"):
            cv2.imwrite(image_path, img)
            print(f"Image saved to '{image_path}'")
            break

        elif key == ord("x"):
            print("Exiting without saving.")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
