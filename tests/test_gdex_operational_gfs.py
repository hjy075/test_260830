import pandas as pd
import requests
from research.topic_lock.gdex_operational_gfs import build_ncss_url, cycle_and_leads, local_horizon_to_utc, MAPPING_VARIANTS, _request_with_retries

def test_august_local_horizon_converts_to_edt_utc():
    target=local_horizon_to_utc(pd.Timestamp('2017-08-14 00:00:00')); assert target[0]==pd.Timestamp('2017-08-14 04:00:00'); assert target[-1]==pd.Timestamp('2017-08-21 03:00:00')
def test_december_local_horizon_converts_to_est_utc():
    target=local_horizon_to_utc(pd.Timestamp('2017-12-25 00:00:00')); assert target[0]==pd.Timestamp('2017-12-25 05:00:00'); assert target[-1]==pd.Timestamp('2018-01-01 04:00:00')
def test_cycle_uses_safe_previous_18z_and_brackets_dst_horizon():
    cycle,leads=cycle_and_leads(pd.Timestamp('2017-08-14 00:00:00')); assert cycle==pd.Timestamp('2017-08-13 18:00:00'); assert leads[0]==9; assert leads[-1]==177
def test_cycle_leads_bracket_est_horizon():
    cycle,leads=cycle_and_leads(pd.Timestamp('2017-12-25 00:00:00')); assert cycle==pd.Timestamp('2017-12-24 18:00:00'); assert leads[0]==9; assert leads[-1]==180
def test_ncss_url_uses_working_classic_netcdf_output():
    url=build_ncss_url(pd.Timestamp('2017-08-13 18:00:00'),9); assert '/d084001/2017/20170813/' in url; assert 'gfs.0p25.2017081318.f009.grib2' in url; assert 'Temperature_height_above_ground' in url; assert 'vertCoord=2' in url; assert 'accept=netcdf' in url; assert 'accept=netcdf4' not in url
def test_mapping_sensitivity_contains_weighted_and_single_variants():
    assert set(MAPPING_VARIANTS)>={'historical_weighted','single_station'}; assert MAPPING_VARIANTS['historical_weighted']['CT']!=MAPPING_VARIANTS['single_station']['CT']; assert MAPPING_VARIANTS['historical_weighted']['WCMASS']!=MAPPING_VARIANTS['single_station']['WCMASS']
def test_transient_read_timeout_is_retried():
    class Response:
        status_code=200; content=b'x'*256; text=''
        def raise_for_status(self): return None
    class Session:
        def __init__(self): self.calls=0
        def get(self,url,timeout):
            self.calls+=1
            if self.calls==1: raise requests.exceptions.ReadTimeout('transient')
            return Response()
    s=Session(); r=_request_with_retries(s,'https://example.invalid',attempts=2,timeout=1,sleep_fn=lambda _:None); assert r.status_code==200; assert s.calls==2
