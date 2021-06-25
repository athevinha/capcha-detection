import requests
import time
# crawler data
url = 'https://www.vietcombank.com.vn/IBanking20/Captcha/JpegImage.ashx?code=goaamamiaiyw&ts=102032'
for i in range(1000):
    r = requests.get(url, allow_redirects=True)
    open('./DATA_TEST/%s.png' % i, 'wb').write(r.content)
    print('Download:', i, ".png...")
    time.sleep(1)
