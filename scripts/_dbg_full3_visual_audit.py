"""Extract existing full3 frames for a read-only simulation audit."""

from pathlib import Path

import cv2


def main():
    video_dir = Path(__file__).resolve().parents[1] / "videos"
    output_dir = video_dir / "full3_rim_mix_audit"
    output_dir.mkdir(exist_ok=True)
    source = video_dir / "mf18_pbstf_pour_full3_closeup.mp4"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    for time_s in (10.5, 12.0, 16.0, 24.0, 27.2, 30.0):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(time_s * fps))
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Cannot decode frame at {time_s}s")
        output = output_dir / f"close_{time_s:04.1f}.jpg"
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"Cannot write {output}")
        print(output)
    capture.release()


if __name__ == "__main__":
    main()
