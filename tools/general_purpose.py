import os

def collect_all_files(target_dir):
    filenames = []
    for path, subdirs, files in os.walk(target_dir):
        for name in files:
            filenames.append(os.path.join(path, name))

    return filenames
