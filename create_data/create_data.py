import random
from pytube import YouTube
import cv2
from typing import Tuple, List
import os
import tempfile
import pathlib
import uuid
import video_database
import matplotlib.pyplot as plt

def detect_faces(img):
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


class UnexistingStreamException(Exception):
    pass

class DataCreator:
    """
    Downloads videos, filters them, and writes the output frames to the data
    :return the fps of the downloaded video
    """

    def _download_video(self, link: str, target_dir: str, filename: str):
        video = YouTube(link)
        streams = video.streams.filter(resolution=self._download_resolution)
        if len(streams) == 0:
            raise UnexistingStreamException(link)

        # find mp4 stream
        stream_to_download = None
        for s in streams:
            if 'mp4' in s.mime_type:
                stream_to_download = s
                break

        if stream_to_download is None:
            raise UnexistingStreamException(link)

        video_fps = stream_to_download.fps
        stream_to_download.download(output_path=target_dir, filename=filename)
        return video_fps

    def _validate_faces(self, frame):
        """
        Check that image contains a single face, and if so returns its center
        :param frame:
        :return: int, tuple(int, int) - # number of faces detected, and if a single face was detected its center
        """

        faces = detect_faces(frame)
        faces_detected = len(faces)
        if faces_detected != 1:
            return faces_detected, None

        x, y, w, h = faces[0]
        return 1, (x + w/2, y + h/2)

    def _process_frame(self, frame):
        faces_detected, center = self._validate_faces(frame)
        if faces_detected != 1:
            return False, None

        # crop around face center
        def get_best_start_finish(center, image_size, crop_size):
            if center < crop_size/2:
                return 0, crop_size

            if image_size - center < crop_size/2:
                return image_size-crop_size, image_size

            return int(center - crop_size/2), int(center + crop_size/2)

        y = get_best_start_finish(center[1], frame.shape[0], self._resolution[1])
        x = get_best_start_finish(center[0], frame.shape[1], self._resolution[0])

        final_frame = frame[y[0]:y[1], x[0]:x[1]]
        return True, final_frame


    def _filter_video(self, source_video, target_dir, fps):
        cap = cv2.VideoCapture(source_video)
        number_of_frames = 0
        while number_of_frames < self._max_frames_per_video and cap.isOpened():
            ret, frame = cap.read()
            success, final_frame = self._process_frame(frame)
            if success:
                cv2.imwrite(os.path.join(target_dir, f"{number_of_frames}.png"), final_frame)
                number_of_frames += 1

            for i in range(int(fps * self._frame_skip)):
                ret, frame = cap.read()

        cap.release()

    def _process_videos(self, links, output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, link in enumerate(links):
                filename = f'{uuid.uuid4()}.mp4'
                try:
                    video_fps = self._download_video(link, target_dir=temp_dir, filename=filename)
                except UnexistingStreamException:
                    continue
                video_out_dir = os.path.join(output_dir, str(index))
                os.makedirs(video_out_dir, exist_ok=True)
                try:
                    self._filter_video(os.path.join(temp_dir, filename), video_out_dir, video_fps)
                except cv2.error:
                    continue


    def __init__(self, frame_skip: float, resolution: Tuple[int, int], max_frames_per_video: int, download_resolution='480p', zoom_on_face=True):
        """
        :param frame_skip: by how much should we skip between frames, in seconds. Used to make sure
               that the data doesn't contain frames that are similar.
        :param resolution: the resolution of the output
        :param max_frames_per_video: the maximum number of frames that are outputted
               per video. Used to make sure our database isn't to big
        :param download_resolution: what youtube resolution should be downloaded
        :param zoom_on_face: should the downloaded video just be resized (false) or should the images
               be cropped to the person's face
        """

        self._frame_skip = frame_skip
        self._resolution = resolution
        self._max_frames_per_video = max_frames_per_video
        self._download_resolution = download_resolution

    def create_data(self, video_links: List[str], val_split=0.2, test_split=0.05):
        """
        Downloads all the videos and filters them.
        :param video_links: the youtube videos to process
        :param val_split: the percentage of videos that will be part of the validation dataset
        :param test_split: the percentage of videos that will be part of the test dataset
        """

        # split videos
        videos = video_links.copy()
        val_links = random.sample(videos, int(val_split * len(video_links)))
        [videos.remove(v) for v in val_links]
        test_links = random.sample(videos, int(test_split * len(video_links)))
        [videos.remove(v) for v in test_links]

        # create dirs
        current_path = pathlib.Path(__file__).parent.resolve()
        data_path = os.path.join(current_path, os.pardir, 'data')
        train_path = pathlib.Path(os.path.join(data_path, 'train'))
        val_path = pathlib.Path(os.path.join(data_path, 'val'))
        test_path = pathlib.Path(os.path.join(data_path, 'test'))

        train_path.mkdir(exist_ok=True)
        val_path.mkdir(exist_ok=True)
        test_path.mkdir(exist_ok=True)

        self._process_videos(videos, train_path)
        self._process_videos(val_links, val_path)
        self._process_videos(test_links, test_path)


if __name__ == '__main__':
    dc = DataCreator(frame_skip=3, resolution=(336, 336), max_frames_per_video=150)
    dc.create_data(video_database.videos)
