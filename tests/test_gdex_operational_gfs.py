# trigger public Actions after workflow registration
import pandas as pd
import requests
from research.topic_lock.gdex_operational_gfs import (
    build_ncss_url,
    cycle_and_leads,
    local_horizon_to_utc,
    MAPPING_VARIANTS,
    _request_with_retries,
    build_osdf_url,
    _find_2t_header_in_bytes,
    _request_range_with_retries,
)


def test_august_local_horizon_converts_to_edt_utc():
    target = local_horizon_to_utc(pd.Timestamp('2017-08-14 00:00:00'))
    assert target[0] == pd.Timestamp('2017-08-14 04:00:00')
    assert target[-1] == pd.Timestamp('2017-08-21 03:00:00')


def test_december_local_horizon_converts_to_est_utc():
    target = local_horizon_to_utc(pd.Timestamp('2017-12-25 00:00:00'))
    assert target[0] == pd.Timestamp('2017-12-25 05:00:00')
    assert target[-1] == pd.Timestamp('2018-01-01 04:00:00')


def test_cycle_uses_safe_previous_18z_and_brackets_dst_horizon():
    cycle, leads = cycle_and_leads(pd.Timestamp('2017-08-14 00:00:00'))
    assert cycle == pd.Timestamp('2017-08-13 18:00:00')
    assert leads[0] == 9
    assert leads[-1] == 177


def test_cycle_leads_bracket_est_horizon():
    cycle, leads = cycle_and_leads(pd.Timestamp('2017-12-25 00:00:00'))
    assert cycle == pd.Timestamp('2017-12-24 18:00:00')
    assert leads[0] == 9
    assert leads[-1] == 180


def test_ncss_url_uses_working_classic_netcdf_output():
    url = build_ncss_url(pd.Timestamp('2017-08-13 18:00:00'), 9)
    assert '/d084001/2017/20170813/' in url
    assert 'gfs.0p25.2017081318.f009.grib2' in url
    assert 'Temperature_height_above_ground' in url
    assert 'vertCoord=2' in url
    assert 'accept=netcdf' in url
    assert 'accept=netcdf4' not in url


def test_osdf_url_preserves_exact_cycle_and_lead():
    url = build_osdf_url(pd.Timestamp('2017-08-13 18:00:00'), 177)
    assert url == (
        'https://osdf-data.gdex.ucar.edu/ncar/gdex/d084001/2017/20170813/'
        'gfs.0p25.2017081318.f177.grib2'
    )


def test_mapping_sensitivity_contains_weighted_and_single_variants():
    assert set(MAPPING_VARIANTS) >= {'historical_weighted', 'single_station'}
    assert MAPPING_VARIANTS['historical_weighted']['CT'] != MAPPING_VARIANTS['single_station']['CT']
    assert MAPPING_VARIANTS['historical_weighted']['WCMASS'] != MAPPING_VARIANTS['single_station']['WCMASS']


def test_transient_read_timeout_is_retried():
    class Response:
        status_code = 200
        content = b'x' * 256
        text = ''
        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.calls = 0
        def get(self, url, timeout):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ReadTimeout('transient')
            return Response()

    s = Session()
    r = _request_with_retries(s, 'https://example.invalid', attempts=2, timeout=1, sleep_fn=lambda _: None)
    assert r.status_code == 200
    assert s.calls == 2


def test_transient_http_504_is_retried():
    class Response:
        content = b'x' * 256
        text = 'gateway timeout'
        def __init__(self, status_code):
            self.status_code = status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f'{self.status_code} error', response=self)

    class Session:
        def __init__(self):
            self.calls = 0
        def get(self, url, timeout):
            self.calls += 1
            return Response(504 if self.calls == 1 else 200)

    s = Session()
    r = _request_with_retries(s, 'https://example.invalid', attempts=2, timeout=1, sleep_fn=lambda _: None)
    assert r.status_code == 200
    assert s.calls == 2


def _section(number, payload=b''):
    length = 5 + len(payload)
    return length.to_bytes(4, 'big') + bytes([number]) + payload


def _fake_grib_message(*, category, number, surface_type, level, total_len=256):
    # Section 4 offsets used by the production parser: template@7:9, cat@9,
    # number@10, surface type@22, scale@23, scaled value@24:28.
    section1 = _section(1, b'\x00' * 16)
    section3 = _section(3, b'\x00' * 20)
    p4 = bytearray(40)
    p4[2:4] = (0).to_bytes(2, 'big')
    p4[4] = category
    p4[5] = number
    p4[17] = surface_type
    p4[18] = 0
    p4[19:23] = int(level).to_bytes(4, 'big')
    section4 = _section(4, bytes(p4))
    prefix = b'GRIB' + b'\x00\x00' + bytes([0, 2]) + int(total_len).to_bytes(8, 'big')
    body = prefix + section1 + section3 + section4
    return body + b'X' * max(0, total_len - len(body))


def test_find_2t_header_in_middle_of_range_chunk():
    other = _fake_grib_message(category=2, number=2, surface_type=103, level=10, total_len=220)
    target = _fake_grib_message(category=0, number=0, surface_type=103, level=2, total_len=462642)
    chunk_start = 10_000_000
    chunk = b'junk-prefix' + other + b'gap' + target[:2048] + b'trailing'
    found = _find_2t_header_in_bytes(chunk, chunk_start)
    expected_offset = chunk_start + len(b'junk-prefix') + len(other) + len(b'gap')
    assert found == (expected_offset, 462642)


def test_range_retry_handles_chunked_encoding_error():
    class Response:
        status_code = 206
        content = b'GRIB'
        headers = {'Content-Range': 'bytes 0-3/100'}
        url = 'https://cache.invalid/file'
        text = ''
        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.calls = 0
        def get(self, url, headers, timeout):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ChunkedEncodingError('incomplete range')
            return Response()

    s = Session()
    r = _request_range_with_retries(
        s,
        'https://example.invalid/file',
        0,
        3,
        attempts=2,
        timeout=1,
        sleep_fn=lambda _: None,
    )
    assert r.status_code == 206
    assert r.content == b'GRIB'
    assert s.calls == 2
