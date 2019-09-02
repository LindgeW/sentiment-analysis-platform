import requests
import sys

def send_post(url, data):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36'
    }
    resp = requests.post(url, data=data, headers=headers)
    if resp.status_code == requests.codes.ok:
        result = resp.json()  # dict
        return result
    else:
        print('error')
        sys.exit()


if __name__ == '__main__':
    res = send_post('http://httpbin.org/post', {'name': 'wlz', 'age': 21})
    print(res)