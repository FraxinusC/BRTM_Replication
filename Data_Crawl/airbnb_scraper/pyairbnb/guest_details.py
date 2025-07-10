from curl_cffi import requests
import json
import base64

def get(api_key: str, cookies: dict, guest_id: str, language: str = "en", proxy_url: str = None):
    # Airbnb 使用 base64 编码的 user ID，例如 User:12345 -> base64
    guest_node = f"User:{guest_id}"
    user_id = base64.b64encode(guest_node.encode()).decode('utf-8')

    # 请求头
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Airbnb-Api-Key": api_key,
    }

    # GraphQL 请求参数
    params = {
        'operationName': 'GetUserProfile',
        'locale': language,
        'currency': 'USD',
        'variables': json.dumps({
            "userId": user_id,
            "isPassportStampsEnabled": True,
            "mockIdentifier": None,
            "fetchCombinedSportsAndInterests": True
        }),
        'extensions': json.dumps({
            "persistedQuery": {
                "version": 1,
                "sha256Hash": "a56d8909f271740ccfef23dd6c34d098f194f4a6e7157f244814c5610b8ad76a"
            }
        })
    }

    url = "https://www.airbnb.com/api/v3/UserProfile"
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    # 发起请求
    response = requests.get(
        url,
        headers=headers,
        cookies=cookies,
        params=params,
        proxies=proxies,
        impersonate="chrome110"
    )

    # 如果出错，抛出异常
    response.raise_for_status()

    return response.json()
