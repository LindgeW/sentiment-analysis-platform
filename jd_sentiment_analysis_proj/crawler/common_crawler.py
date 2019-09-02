import requests
import os
import re
import json
import time


def save_to_txt(file_dir, file_name, comms):
    if len(comms) != 0:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        path = os.path.join(file_dir, file_name)
        with open(path, 'a', encoding='utf8') as fout:
            for com in comms:
                fout.write("{}\n".format(com))


def crawl_comm(product_id, score, page_id):
    url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv20226&productId=%s&score=%d&sortType=5&page=%d&pageSize=10&isShadowSku=0&fold=1' % (
    product_id, score, page_id)
    # url = 'https://sclub.jd.com/comment/productPageComments.action?productId=%s&score=%d&sortType=5&page=%d&pageSize=10&isShadowSku=0&fold=1'%(product_id, score, page_id)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code != requests.codes.ok:
            return

        doc = resp.text
        if doc is None or doc == '':
            print('No response')
            return

        if 'callback=fetchJSON_comment' in url:
            # 对整个字符串进行搜索匹配，返回第一个匹配的字符串的 match 对象
            match = re.search(re.compile(r'.*?\((.*)\);', re.S | re.M), doc)
            doc = match.group(1)  # 提取json字符串

        comm_set = set()
        json_data = json.loads(doc, strict=False)  # json字符串转字典
        for com in json_data['comments']:
            product_name = com['referenceName']  # 产品名
            # com_time = com['referenceTime'] #评论时间
            comment = com['content']  # 评论内容
            if len(comment.strip()) > 8 and not '此用户未填写评价内容' in comment:
                # item = '{},{},{},{}'.format(product_id, re.sub('\s', '', comment), score, com_time)
                item = '{},{}'.format(re.sub('[\s]', '', comment), score)
                comm_set.add(item)

        if len(comm_set) != 0:
            if score >= 4:
                file_name = 'pos_comments.txt'
            elif score <= 1:
                file_name = 'neg_comments.txt'
            else:
                file_name = 'gen_comments.txt'
            save_to_txt(product_name, file_name, comm_set)

    except Exception as e:
        print("Unexpected error:", e)
        return


if __name__ == '__main__':
    print('start...')
    product_id = input("输入商品ID：").strip()
    if product_id != '':
        t1 = time.time()
        for i in range(40):
            for score in range(3, 0, -1):  # 3 2 1
                crawl_comm(product_id, score, i)
                time.sleep(1)
            time.sleep(1)
        t2 = time.time()
        print('总用时：%.2f'%(t2 - t1))
