from inplace.meta_client import should_retry


def test_retry_only_rate_limits_and_server_errors() -> None:
    assert should_retry(429)
    assert should_retry(500)
    assert should_retry(503)
    assert not should_retry(400)
    assert not should_retry(404)
    assert not should_retry(200)
