import urllib.request
import re

url = "https://itviec.com/it-jobs"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    links = set(re.findall(r'href="(/it-jobs/[a-zA-Z0-9\-]+)"', html))
    print("ITviec links:")
    for link in list(links)[:3]:
        print(f"https://itviec.com{link}")
except Exception as e:
    print(e)

url2 = "https://www.topcv.vn/viec-lam"
req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html2 = urllib.request.urlopen(req2).read().decode('utf-8')
    links2 = set(re.findall(r'href="(https://www.topcv.vn/viec-lam/[a-zA-Z0-9\-]+\.html\?.*?)', html2))
    print("\nTopCV links:")
    for link in list(links2)[:7]:
        print(link)
except Exception as e:
    print(e)
