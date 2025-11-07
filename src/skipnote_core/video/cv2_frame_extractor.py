import cv2
from typing import List, override
from skipnote_core.video.frame import Frame, InMemoryFrame
from skipnote_core.video.frame_extractor import VideoFrameExtractor


class CV2FrameExtractor(VideoFrameExtractor):

    @override
    def extract_frames(self, video_path: str, frame_interval: int, initial_offset: int = 0, **kwargs) -> List[Frame]:
        
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frame_count / fps

        extracted_frames: List[Frame] = []
        current_time_sec = initial_offset

        while current_time_sec <= video_duration:
            # Seek to the target time (in milliseconds)
            video_capture.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)
            success, frame_data = video_capture.read()
            if not success:
                break

            extracted_frames.append(
                InMemoryFrame(timestamp=current_time_sec, data=cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB))
            )

            current_time_sec += frame_interval

        video_capture.release()
        return extracted_frames
    

if __name__ == "__main__":
    video_path = "/home/nricciardi/Repositories/skipnote/src/skipnote_core/video/video.mp4"
    extractor = CV2FrameExtractor()
    frames = extractor.extract_frames(video_path, frame_interval=60)
    print(f"Extracted {len(frames)} frames from the video.")

    frames[0].save("/home/nricciardi/Repositories/skipnote/src/skipnote_core/video/first_extracted_frame.jpg")
    print(frames[0].timestamp)
    print(frames[1].timestamp)

