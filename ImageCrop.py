import io
import os
import os.path

from PIL import Image


def ImageCrop(imagePath, size, saveFolder, start_index=0):
    image = Image.open(imagePath)
    w, h = image.size

    for _w in range(0, w - size + 1, size):
        for _h in range(0, h - size + 1, size):
            croppedImage = image.crop((_w, _h, _w + size, _h + size))
            if os.path.exists(saveFolder + "\\" + str(_w // size * 2 + _h // size)) is False:
                os.makedirs(saveFolder + "\\" + str(_w // size * 2 + _h // size))
            savePath = saveFolder + "\\" + str(_w // size * 2 + _h // size) + "\\" + str(start_index) \
                       + "." + imagePath.split(".")[-1]
            croppedImage.save(savePath)
            start_index += 1

    return start_index


def renameFiles(_pathFolder):
    _fileNames = os.listdir(_pathFolder)

    count = 0
    for _file in _fileNames:
        rename = _pathFolder + "\\" + str(count) + "." + _file.split(".")[-1]
        os.rename(_pathFolder + '\\' + _file, rename)
        count += 1

    print(" Total", count, " Files was renamed")


def checkFiles(_pathFolder):
    error_num = 0
    _fileNames = os.listdir(_pathFolder)

    for _file in _fileNames:
        try:
            print('check Files.. : ', _file)
            Image.open(_pathFolder + "\\" + _file)
        except OSError as e:
            print('>>>> [!] ERROR : ', e.filename)
            error_num += 1

    print('Total Error : ', error_num)


def cropAllImagesInFolder(_pathFolder, _saveFolder, cropSize=3):
    cropSize = cropSize
    _count = 0

    _fileNames = os.listdir(_pathFolder)

    for _file in _fileNames:
        targetPath = _pathFolder + "\\" + _file
        end_index = ImageCrop(targetPath, cropSize, _saveFolder, _count)
        _count = end_index + 1
        print('File ', _file, ' was Cropped!')


# TODO : 이미지가 들어있는 폴더 경로를 수정하세요 => pathFolder
# pathFolder = 'E:\\Fruit_360\\fruits-360_dataset\\fruits-360\\Training\\Apple Crimson Snow'
pathFolder = 'G:\\내 드라이브\\연세대 미래캠\\2학년 1학기\\현승민 선배 논문\\Defense Gan-At-ICON\\adversarial_path\\Apple'

# TODO : Crop된 이미지들을 저장할 폴더 경로를 수정하세요 => saveFolder
# saveFolder = 'E:\\Fruit_360\\fruits-360_dataset\\DefenseGan\\Apple'
saveFolder = 'G:\\내 드라이브\\연세대 미래캠\\2학년 1학기\\현승민 선배 논문\\Defense Gan-At-ICON\\adversarial_path\Apple'

# Usage : 폴더 내부의 모든 이미지에 대해서 처리하는 경우
cropAllImagesInFolder(pathFolder, saveFolder, cropSize=50)

# checkFiles : 폴더 내부의 파일들에 대해서 PIL Image로 열리는지 확인
# - checkFiles(saveFolder)
