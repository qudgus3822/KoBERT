import os


def is_drive_mounted(mount_path="/content/drive"):
    """
    Drive 마운트 경로가 존재하는지 확인하여 접속 가능 여부를 판단합니다.
    """
    # /content/drive 경로가 디렉터리(폴더)로 존재하는지 확인합니다.
    return os.path.isdir(mount_path)


def is_running_in_colab():
    """
    현재 코드가 Google Colab 런타임 환경에서 실행 중인지 확인합니다.
    """
    try:
        # google.colab 모듈이 import 가능한지 시도합니다.
        import google.colab

        # import에 성공하면 Colab 환경입니다.
        return True
    except ImportError:
        # import에 실패하면 Colab 환경이 아닙니다.
        return False


print(is_running_in_colab())
