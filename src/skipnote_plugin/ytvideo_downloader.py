from yt_dlp import YoutubeDL


class YTVideoDownloader:

    @classmethod
    def download_video(cls, url: str, output_path: str) -> str:
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'concurrent_fragment_downloads': 10,
            'noplaylist': True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict)

        return video_path
    

if __name__ == "__main__":
    import os

    YTVideoDownloader.download_video(
        url="https://www.youtube.com/watch?v=JXsQOpGMjU4&list=PLwTFe8oQBgKbRf1NU5NZc_Zmti1O6dt1f&index=2",
        output_path=os.path.join(os.getenv("PYTHONPATH"), "skipnote_plugin/video.mp4")
    )

    print("Download completed.")
