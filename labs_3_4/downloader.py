import requests
import sys
import json
import shutil
import os


def get_dataset_info(owner, repo, path):
    headers = {'Accept': 'application/vnd.github.v3+json'}

    resp = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents/{path}',
                        headers=headers)

    dataset_info = []

    if resp.status_code == 200:
        resp_json = json.loads(resp.text)

        for item in resp_json:
            if item['type'] == 'dir':
                dataset_info.append({
                        'name': item['name'],
                        'url': item['url']
                })

    return dataset_info


def get_files_list(url):
    headers = {'Accept': 'application/vnd.github.v3+json'}

    resp = requests.get(url, headers=headers)

    files_list = []

    if resp.status_code == 200:
        resp_json = json.loads(resp.text)

        for item in resp_json:
            if item['type'] == 'file':
                files_list.append({
                        'name': item['name'].split('?', 1)[0],
                        'download_url': item['download_url'].split('?', 1)[0].split('%', 1)[0]
                })

    return files_list


def download_dataset_subitem(path, url):
    files_list = get_files_list(url)

    total = len(files_list)
    i = 0

    for file in files_list:
        r = requests.get(file['download_url'])
        if r.status_code == 200:
            with open(os.path.join(path, file['name']), 'wb') as f:
                f.write(r.content)

        i += 1
        print('%s: %.2f%%' % (path, i / total * 100.0), end='\r')

    print()


def download_dataset(dataset_info):
    DATASET_DIR = 'dataset'

    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
    else:
        shutil.rmtree(DATASET_DIR)

    for item in dataset_info:
        item_path = os.path.join(DATASET_DIR, item['name'])
        if not os.path.exists(item_path):
            os.makedirs(item_path)

        download_dataset_subitem(item_path, item['url'])


CLASS_SIZE = 170


def main(args):
    owner = args[0]
    repo = args[1]
    path = args[2]

    # dataset_info = get_dataset_info(owner, repo, path)
    # download_dataset(dataset_info)

    train_path = os.path.join('train')

    if not os.path.exists(train_path):
        os.mkdir(train_path)

    class_names = os.listdir('dataset')
    for class_name in class_names:
        class_path = os.path.join('dataset', class_name)
        images = os.listdir(class_path)[:CLASS_SIZE]
        train_class_path = os.path.join(train_path, class_name)

        if not os.path.exists(train_class_path):
            os.mkdir(train_class_path)

        for image in images:
            src_path = os.path.join(class_path, image)
            dst_path = os.path.join(train_class_path, image)
            shutil.copy2(src_path, dst_path)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
