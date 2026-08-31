"""HTTP 계층. 사내망/백신의 TLS 검사(SSL 인터셉트) 환경을 자동 처리한다.

증상: requests 로 위키피디아/FRED/SEC 접속 시
      SSLCertVerificationError: self signed certificate in certificate chain

원인: 백신·방화벽이 TLS 트래픽을 중간에서 다시 서명. 이 루트 CA는 Windows
      인증서 저장소에는 있지만 certifi 번들에는 없다.

해결 순서:
  1) truststore  : OS 인증서 저장소를 그대로 사용 (권장, pip install truststore)
  2) CA_BUNDLE   : config.CA_BUNDLE 또는 환경변수 REQUESTS_CA_BUNDLE 에 pem 경로
  3) VERIFY=False: 최후 수단. 경고를 띄우고 검증을 끈다 (config.VERIFY_SSL=False)
"""
from __future__ import annotations

import os
import time as _time
import warnings

import requests

from . import config as C

_READY = False
_MODE = "default"


def setup() -> str:
    """프로세스 전역 SSL 설정. 반환값은 적용된 모드 이름."""
    global _READY, _MODE
    if _READY:
        return _MODE

    bundle = getattr(C, "CA_BUNDLE", "") or os.environ.get("REQUESTS_CA_BUNDLE", "")
    if bundle and os.path.exists(bundle):
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
        os.environ["SSL_CERT_FILE"] = bundle
        _MODE = f"ca-bundle({bundle})"
    else:
        try:
            import truststore
            truststore.inject_into_ssl()
            _MODE = "truststore"
        except ImportError:
            if getattr(C, "VERIFY_SSL", True):
                _MODE = "default"
            else:
                _MODE = "verify-disabled"

    if _MODE == "verify-disabled":
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass

    _READY = True
    return _MODE


def session(user_agent: str = "Mozilla/5.0 (research script)") -> requests.Session:
    setup()
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    if _MODE == "verify-disabled":
        s.verify = False
    return s


def get(url: str, user_agent: str = "Mozilla/5.0 (research script)",
        retries: int = 3, **kw):
    """단발 GET. 타임아웃/일시적 오류는 지수 백오프로 재시도."""
    s = session(user_agent)
    kw.setdefault("timeout", 90)
    last = None
    for attempt in range(retries):
        try:
            r = s.get(url, **kw)
            r.raise_for_status()
            return r
        except requests.exceptions.SSLError as e:
            last = e
            break
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            last = e
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__} -> {wait}s 대기")
                _time.sleep(wait)
    try:
        raise last
    except requests.exceptions.SSLError as e:
        raise RuntimeError(
            f"SSL 검증 실패 ({url}).\n"
            f"  현재 모드: {_MODE}\n"
            f"  해결 1: pip install truststore   (가장 간단)\n"
            f"  해결 2: 사내 루트 CA를 pem 으로 내보내 config.CA_BUNDLE 에 경로 지정\n"
            f"  해결 3: config.VERIFY_SSL = False  (보안상 권장하지 않음)\n"
            f"  원인: {e}"
        ) from e
