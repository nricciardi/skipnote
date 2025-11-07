from typing import List
from abc import ABC, abstractmethod
from skipnote_core.video.frame import Frame


class VideoFrameExtractor(ABC):

    @abstractmethod
    def extract_frames(self, video_path: str, frame_interval: int, initial_offset: int = 0, **kwargs) -> List[Frame]:
        return NotImplemented