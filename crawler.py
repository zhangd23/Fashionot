from icrawler.builtin import GoogleImageCrawler
import os

# Root directory 
path = os.path.dirname(os.path.realpath('crawler.py'))

# Folder paths
folder1 = path+r'\images\fashionable'
folder2 = path+r'\images\unfashionable'

# Create Folders
if not os.path.exists(folder1):
    os.makedirs(folder1)

if not os.path.exists(folder2):
    os.makedirs(folder2)

# Positive results parser
google_crawler = GoogleImageCrawler(
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': folder1}
)

for keyword in ['mens fashion', 'mens outfit']:
    google_crawler.crawl(
        keyword=keyword, max_num=10, min_size=(500, 700),max_size=(800,1000), file_idx_offset='auto')
    # set `file_idx_offset` to 'auto' will prevent naming the 5 images
    # of dog from 000001.jpg to 000005.jpg, but naming it from 000006.jpg.

# Negative results parser
google_crawler = GoogleImageCrawler(
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': folder2}
)

for keyword in ['mens hideous outfit']:
    google_crawler.crawl(
        keyword=keyword, max_num=10, min_size=(500, 700),max_size=(800,1000), file_idx_offset='auto')
    # set `file_idx_offset` to 'auto' will prevent naming the 5 images
    # of dog from 000001.jpg to 000005.jpg, but naming it from 000006.jpg.
