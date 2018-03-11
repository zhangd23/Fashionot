from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': '/home/kivy/code/Fashionot/images/'}
)

for keyword in ['mens fashion', 'mens outfit']:
    google_crawler.crawl(
        keyword=keyword, max_num=10000, min_size=(500, 700),max_size=(800,1000), file_idx_offset='auto')
    # set `file_idx_offset` to 'auto' will prevent naming the 5 images
    # of dog from 000001.jpg to 000005.jpg, but naming it from 000006.jpg.
