from pyquery import PyQuery as pq
import re


def test1():
    html = '''
    <div class="wrap">
    <div id="container">
    <ul class ="list">
        <li class="item-0">first item</li>
        <li class ="item-1">
            <a href="//item.jd.com/link2.html">second item</a>
        </li>
        <li class="item-0 active">
            <a href="//item.jd.com/link3.html"><span class="bold">third item</span></a>
        </li>
        <li class ="item-1 active">
            <a href="//item.jd.com/link4.html">fourth item</a>
        </li>
        <li class="item-0">
            <a href="//item.jd.com/links.html">fifth item</a>
        </li>
    </ul>
    </div>
    </div>
    '''

    doc = pq(html)

    # for item in doc('a').items():
    #     link = item.attr('href')
    #     if re.match(r'link\d', link):
    #         print(link)

    if doc('a'):
        print(doc('a').make_links_absolute(base_url='https://item.jd.com/123.html#comment'))

    # z = filter(lambda x: re.match(r'link\d', x.attr('href')), doc('a').items())
    # for i in z:
    #     print(i.attr('href'))
