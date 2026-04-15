import cv2

def test_camera(index):
    print(f"Testing /dev/video{index}...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Failed to open /dev/video{index}")
        return
    
    ret, frame = cap.read()
    if ret:
        filename = f"webcam_test_{index}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Successfully saved image as {filename}")
    else:
        print(f"Failed to read frame from /dev/video{index}")
    cap.release()

# ログで有望だった 2, 4, 6 をテスト
for i in [2, 4, 6]:
    test_camera(i)