# webmagic改版 - 通用爬虫
import requests
# from urllib import request
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin


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
    def download(self, url):
        if url is None:
            return None
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        }
        # requests请求一定要加上headers，好让访问的服务器知道这是一个正常的浏览器操作
        response = requests.get(url, headers=headers)  # 发送GET请求
        if response.status_code != requests.codes.ok:  # 200
            return None
        return response.content.decode('utf8')


# 网页解析器
class HtmlParser(object):
    def _get_new_urls(self, page_url, soup):
        new_urls = set()
        # [\u4e00-\u9fa5] 判断字符串中是否包含中文
        #         links = soup.findAll('a', href=re.compile(r"/item/^[\w\u4e00-\u9fa5]+$/\d{0,}"))
        links = soup.findAll('a', href=re.compile(r"^/item/(.*?)/\d{0,}$"))
        for link in links:
            new_url = link['href']  # 获取链接的href属性
            new_full_url = urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self, page_url, soup):
        res_data = dict()  # 以字典形式保存数据
        res_data['url'] = page_url
        # <dd class="lemmaWgt-lemmaTitle-title"><h1>Python</h1>
        title_node = soup.find('dd', class_="lemmaWgt-lemmaTitle-title").find("h1")
        res_data['title'] = title_node.get_text()
        # <div class="lemma-summary" label-module="lemmaSummary">
        summary_node = soup.find('div', class_="lemma-summary")
        res_data['summary'] = summary_node.get_text()
        return res_data

    def parse(self, page_url, html_doc):
        if page_url is None or html_doc is None:
            return
        soup = BeautifulSoup(html_doc, 'lxml')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_data(page_url, soup)

        return new_urls, new_data


# 数据输出
class HtmlOutputer(object):
    def __init__(self):
        self.datas = []

    def collect_data(self, data):
        if data is None:
            return
        self.datas.append(data)

    def output_html(self, filename):  # 将抓取到的数据以表格形式输出显示
        with open(filename, 'w', encoding='utf-8') as fout:
            fout.write("<html>")
            fout.write('<head><meta charset="utf-8"></head>')
            fout.write("<body>")
            fout.write("<table border='1'>")
            fout.write("<tr><th>编号</th><th>URL</th><th>标题</th><th>摘要</th></tr>")
            for i, data in enumerate(self.datas):
                fout.write("<tr>")
                fout.write("<td>{0}</td>".format(i + 1))
                fout.write("<td>{0}</td>".format(data['url']))
                fout.write("<td>{0}</td>".format(data['title']))
                fout.write("<td>{0}</td>".format(data['summary']))
                fout.write("</tr>")

            fout.write("</table>")
            fout.write("</body>")
            fout.write("</html>")


# 调度器
class SpiderMain(object):
    def __init__(self):
        self.urls = UrlManager()
        self.downloader = HtmlDownloader()
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
                new_urls, new_data = self.parser.parse(new_url, html_doc)  # 解析网页内容，返回新URL和数据
                self.urls.add_new_urls(new_urls)  # 将新URL添加到URL管理器中
                self.outputer.collect_data(new_data)  # 收集新的数据
                if count == nb_url:
                    break

                count += 1
            except Exception as e:
                print("crawl failed!!!", e)

        self.outputer.output_html('output.html')


if __name__ == "__main__":
    # 爬取百度百科python页的n条网页数据
    root_url = "https://baike.baidu.com/item/Python/407313"
    obj_spider = SpiderMain()
    obj_spider.crawl(root_url, 30)
