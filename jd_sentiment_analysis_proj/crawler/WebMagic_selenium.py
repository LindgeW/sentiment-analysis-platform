# webmagic改版 - 通用爬虫
# import requests
# from bs4 import BeautifulSoup
from pyquery import PyQuery as pq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import time


# URL管理器
class UrlManager(object):
    def __init__(self):
        self.new_urls = set()  # 待爬取的链接集
        self.old_urls = set()  # 已爬取的链接集

    def add_new_url(self, url):  # 添加一个URL
        if url is None:
            return
        # 全新的URL保存到待爬取的链接集
        if url not in self.new_urls and url not in self.old_urls:
            self.new_urls.add(url)  # 入队

    def add_new_urls(self, urls):  # 批量添加URL
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)

    def has_new_url(self):  # 还有待爬取的URL
        return len(self.new_urls) != 0

    def get_new_url(self):  # 获取一个待爬取的URL，并移除
        new_url = self.new_urls.pop()  # 出队
        self.old_urls.add(new_url)
        return new_url


# 网页下载器(整合selenium抓动态页面)
class HtmlDownloader(object):
    def __init__(self, driver_path):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.browser = webdriver.Chrome(options=chrome_options, executable_path=driver_path)
        self.browser.delete_all_cookies()  # 清除浏览器cookies
        self.browser.implicitly_wait(3)  # 隐式等待：对整个driver的周期都起作用，所以只要设置一次即可

    def download(self, url):
        if url is None:
            return None

        doc = None
        try:
            self.browser.get(url)  # 请求网页

            # print(self.browser.current_url, self.browser.current_window_handle)
            # 显式等待：最长等待时间10s
            WebDriverWait(self.browser, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.comment-con')))
            # WebDriverWait(self.browser, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.comment-con')))

            # time.sleep(2)

            # pyquery默认解析器是xml类型
            doc = pq(self.browser.page_source, parser='html')  # 页面源码
            # soup = BeautifulSoup(self.browser.page_source, "lxml")
        except Exception as e:  # 可以捕获除与程序退出sys.exit()相关之外的所有异常
            print('exception occurred!', e)
        # finally:
        #     self.browser.close()   # 关闭当前窗口
        #     self.browser.quit()  # 退出驱动并关闭所有关联的窗口
        return doc


# 网页解析器
class HtmlParser(object):
    def _get_new_urls(self, page_url, doc):
        if doc is None:
            return

        new_urls = set()
        for item in doc('a').make_links_absolute(base_url=page_url).items():
            url = item.attr('href')
            if isinstance(url, str) and re.match(r'https://item.jd.com/\d+.html', url):
                new_urls.add(url)
            else:
                print('invalid url：', url)

        return new_urls

    def _get_new_data(self, doc):
        # res_data = dict()  # 以字典形式保存数据
        res_data = set()
        items = doc('#comment-0 .comment-item .comment-column').items()
        for item in items:
            comment = re.sub(r'\s', '', item.find('.comment-con').text())  # 去掉评论中的空白字符
            star = re.sub(r'\D+', '', item.find('.comment-star').attr('class'))  # 去掉星级中的非数字字符
            res_data.add((star, comment))
        return res_data

    def parse(self, page_url, doc):
        if doc is None:
            return
        new_urls = self._get_new_urls(page_url, doc)
        new_data = self._get_new_data(doc)
        return new_urls, new_data


# 数据输出
class HtmlOutputer(object):
    def __init__(self):
        self.datas = []

    def collect_data(self, data):
        if data is None:
            return
        print('data: ', data)
        self.datas.append(data)

    def output_html(self, filename):  # 将抓取到的数据输出文件
        with open(filename, 'w', encoding='utf-8') as fout:
            for set_data in self.datas:
                for pair in set_data:
                    fout.write('%s\t%s\n' % pair)


# 调度器
class SpiderMain(object):
    def __init__(self, driver_path):
        self.urls = UrlManager()
        self.downloader = HtmlDownloader(driver_path)
        self.parser = HtmlParser()
        self.outputer = HtmlOutputer()

    def crawl(self, root_url, nb_url):
        count = 1
        self.urls.add_new_url(root_url)  # 初始化URL管理器（相当于队列）
        while self.urls.has_new_url:  # URL管理器中是否有URL?
            try:
                new_url = self.urls.get_new_url()  # 取出新URL
                print("crawled {0} : {1}".format(count, new_url))
                html_doc = self.downloader.download(new_url)  # 下载对应的网页
                new_urls, new_data = self.parser.parse(root_url, html_doc)  # 解析网页内容，返回新URL和数据
                self.urls.add_new_urls(new_urls)  # 将新URL添加到URL管理器中
                self.outputer.collect_data(new_data)  # 收集新的数据
            except Exception as e:
                print("crawl failed!!!", e)

            if count == nb_url:
                break
            count += 1

        self.outputer.output_html('out.txt')


if __name__ == "__main__":
    # 爬取京东评论数据
    root_url = "https://item.jd.com/100003438286.html#comment"
    # 也可以直接将chromedriver.exe放到Scripts目录下
    obj_spider = SpiderMain(driver_path='F:/driver/chromedriver.exe')
    obj_spider.crawl(root_url, nb_url=10)
