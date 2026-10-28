import requests
url = "https://docs.djangoproject.com/en/4.0/ref/models/overview"
r = requests.get(url, timeout=10, allow_redirects=False)
print(r.status_code, repr(r.headers.get("Location")), len(r.text))
