from typing import List
from argparse import ArgumentParser
from os import listdir, path
from subprocess import Popen, check_output


def look_for_files(folder: str) -> List[str]:
    return [path.abspath(path.join(folder, filename))
            for filename in listdir(path=folder)
            if path.isfile(path.join(folder, filename)) and
            filename.rsplit('.', maxsplit=1)[1].lower() in [
                "webm",
                "avi",
                "mp4",
                "mov",
                "mkv"
            ]]


def ffprobe_get_vcodec(filename: str) -> str:
    try:
        return str(check_output('ffprobe -v error -select_streams v:0 -show_entries stream=codec_name ' +
                                '-of default=noprint_wrappers=1:nokey=1 ' +
                                f'"{filename}"',
                                shell=True))
    except OSError as e:
        print("OSError: ", e)
    except ValueError as e:
        print("ValueError: Couldn't call FFMPEG with these parameters\n", e)


def ffprobe_get_acodec(filename: str) -> str:
    try:
        return str(check_output('ffprobe -v error -select_streams a:0 -show_entries stream=codec_name ' +
                                '-of default=noprint_wrappers=1:nokey=1 ' +
                                f'"{filename}"',
                                shell=True))
    except OSError as e:
        print("OSError: ", e)
    except ValueError as e:
        print("ValueError: Couldn't call FFMPEG with these parameters\n", e)


def ffmpeg_convert(filename: str, out_path) -> None:
    out_filename = path.abspath(path.join(out_path, path.split(filename)[-1].rsplit('.', maxsplit=1)[0] + ".mov"))
    if path.isfile(out_filename):
        return
    print(ffprobe_get_vcodec(filename), ffprobe_get_acodec(filename))
    vcodec_option = "-vcodec mjpeg" if ffprobe_get_vcodec(filename) not in (
        "h264",
        "mjpeg"
    ) else "-vcodec copy"
    acodec_option = "-acodec pcm_s16le" if ffprobe_get_acodec(filename) not in (
        "pcm_s16be",
        "pcm_s16le"
    ) else "-acodec copy"
    print("--------------------\nRunning:")
    print(f'ffmpeg -i "{filename}" {vcodec_option} {acodec_option} "{out_filename}"')
    try:
        Popen(f'ffmpeg -i "{filename}" {vcodec_option} {acodec_option} "{out_filename}"',
              universal_newlines=True, shell=True)
    except OSError as e:
        print("OSError: ", e)
    except ValueError as e:
        print("ValueError: Couldn't call FFMPEG with these parameters\n", e)


if __name__ == "__main__":
    ap = ArgumentParser(usage=f"python {__name__} <path_to_files_for_being_converted> <dest_path: default is ..>\n" +
                              "The filenames will stay the same but the file ending will change.")
    ap.add_argument('-i', "--in_path", help="Path to files to be converted", default=".")
    ap.add_argument('-o', "--out_path", help="Folder where the clonverted files will be placed", default="..")
    args = ap.parse_args()
    file_list = look_for_files(args.in_path)
    for file in file_list:
        ffmpeg_convert(file, args.out_path)
