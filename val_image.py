import os
import shutil

val_txt = '/home/cvlab/datasets/PASCAL_MT/ImageSets/Context/val.txt'  # 이미지 목록이 적힌 파일
source_dir = '/home/cvlab/datasets/PASCAL_MT/JPEGImages'  # 원본 이미지가 있는 폴더
dest_dir = '/home/cvlab/datasets/PASCAL_MT/val_images'  # 복사해 둘 폴더 (없으면 생성)

# 0) 목적지 폴더가 없다면 생성
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 1) val.txt에 있는 이미지 이름들을 한 줄씩 읽음
with open(val_txt, 'r') as f:
    lines = f.read().splitlines()

# 2) 각 이미지 이름을 이용해 JPEGImages 폴더에서 해당 파일을 찾아 복사
for img_name in lines:
    # 확장자를 .jpg 로 가정
    filename = img_name + '.jpg'
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(dest_dir, filename)

    # 파일이 실제로 존재하는지 확인 후 복사
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {filename} to {dest_dir}")
    else:
        print(f"Warning: {src_path} does not exist!")