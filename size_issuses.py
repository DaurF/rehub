import os

source_folder = 'frame_from_6_video'
target_folder = 'frame_from_6_video'

os.makedirs(target_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.startswith('person4_') and filename.endswith('.jpg'):
        cnt = filename[len('person_'):-len('.jpg')]

        new_filename = f'person11_{cnt}.jpg'

        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(target_folder, new_filename)

        os.rename(src_path, dst_path)
        print(f'Renamed {src_path} to {dst_path}')