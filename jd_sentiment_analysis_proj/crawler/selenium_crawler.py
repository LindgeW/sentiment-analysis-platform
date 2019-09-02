from selenium import webdriver
from pyquery import PyQuery as pq
# from urllib.parse import quote
import time
import re

'''
selenium操作流程：
1、创建浏览器driver（设置相关参数）
2、传URL，请求网页
3、通过page_source属性获取网页的源代码，接着就可以使用解析库
（如正则表达式、BeautifulSoup、pyquery等）来提取信息
'''


def save_to_txt(path, coms):
    if len(coms) != 0:
        with open(path, 'a', encoding='utf8', errors='ignore') as fout:
            for com in coms:
                fout.write("{}\n".format(com))


class Crawler(object):
    def __init__(self, driver_path):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.browser = webdriver.Chrome(options=chrome_options, executable_path=driver_path)
        self.browser.delete_all_cookies()  # 清除浏览器cookies

    def jd_extract(self, url, page_num=10):
        assert url is not None or url != ''
        # self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight)') #小技巧：下滑滚动条，使评论数据加载出来
        try:
            self.browser.get(url)  # 请求网页
            for i in range(page_num):
                print('crawling Page %d' % (i + 1))
                doc = pq(self.browser.page_source)  # 页面源码
                items = doc('#comment-0 .comment-item .comment-column').items()
                com_set = set()
                for item in items:
                    comment = re.sub(r'\s', '', item.find('.comment-con').text())  # 去掉评论中的空白字符
                    star = re.sub(r'\D+', '', item.find('.comment-star').attr('class'))  # 去掉星级中的非数字字符
                    if len(comment.strip()) > 8 and '此用户未填写评价内容' not in comment:
                        com_set.add(comment.strip())
                if int(star) >= 4:
                    save_to_txt('pos_comments.txt', com_set)
                elif int(star) <= 1:
                    save_to_txt('neg_comments.txt', com_set)
                else:
                    save_to_txt('gen_comments.txt', com_set)

                time.sleep(1)

                # 执行js代码模拟点击下一页,不能用click，因为click点击字符串没用
                # browser.execute_script('document.getElementsByClassName("ui-pager-next")[0].click()')
                self.browser.execute_script('$(".com-table-footer .ui-page .ui-pager-next").click()')
                # self.browser.execute_script('$(".com-table-footer .ui-page .ui-page-curr").next().click()')
        except Exception as e:  # 可以捕获除与程序退出sys.exit()相关之外的所有异常
            print('exception occurred!', e)
        finally:
            self.browser.quit()


if __name__ == '__main__':
    jd_crawler = Crawler(driver_path='F:/driver/chromedriver.exe')
    t1 = time.time()
    print('数据爬取中......')
    jd_crawler.jd_extract(url='https://item.jd.com/100002928171.html#comment')
    t2 = time.time()
    print(t2 - t1)
