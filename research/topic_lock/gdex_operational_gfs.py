from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import xarray as xr

from research.topic_lock.gfc17_seed_null import (
    HORIZON,
    _load_fev_long,
    _supervised,
    exact_cut_indices,
    model_factory,
    safe_feature_columns,
)

NCSS_BASE = 'https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001'
OSDF_BASE = 'https://osdf-data.gdex.ucar.edu/ncar/gdex/d084001'
SOURCE_TZ = 'America/New_York'
BBOX = dict(north=45.6, south=40.5, west=-74.5, east=-68.5)
OSDF_TARGET_RATIO = 0.661
OSDF_FAST_WINDOWS = (2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024)

STATIONS = {
    'BOS': (42.3606, -71.0098),
    'BDR': (41.1635, -73.1262),
    'BTV': (44.4683, -73.1499),
    'CON': (43.2049, -71.5026),
    'PWM': (43.6423, -70.3045),
    'PVD': (41.7223, -71.4325),
    'BDL': (41.9389, -72.6832),
    'ORH': (42.2706, -71.8731),
}

MAPPING_VARIANTS = {
    'historical_weighted': {
        'ME': {'PWM': 1.0},
        'NH': {'CON': 1.0},
        'VT': {'BTV': 1.0},
        'CT': {'BDL': .13, 'BDR': .87},
        'RI': {'PVD': 1.0},
        'SEMASS': {'PVD': 1.0},
        'WCMASS': {'BDL': .5, 'ORH': .5},
        'NEMASSBOST': {'BOS': 1.0},
    },
    'single_station': {
        'ME': {'PWM': 1.0},
        'NH': {'CON': 1.0},
        'VT': {'BTV': 1.0},
        'CT': {'BDL': 1.0},
        'RI': {'PVD': 1.0},
        'SEMASS': {'PVD': 1.0},
        'WCMASS': {'ORH': 1.0},
        'NEMASSBOST': {'BOS': 1.0},
    },
}


def normalize_zone(x):
    s = str(x).upper().replace('-', '').replace('_', '').replace(' ', '')
    for key, zone in [
        ('NEMASSBOST', 'NEMASSBOST'), ('NEMASS', 'NEMASSBOST'), ('NEMA', 'NEMASSBOST'),
        ('SEMASS', 'SEMASS'), ('SEMA', 'SEMASS'), ('WCMASS', 'WCMASS'), ('WCMA', 'WCMASS'),
        ('ME', 'ME'), ('NH', 'NH'), ('VT', 'VT'), ('CT', 'CT'), ('RI', 'RI'),
    ]:
        if key in s:
            return zone
    raise ValueError(x)


def local_horizon_to_utc(forecast_start):
    local = pd.date_range(pd.Timestamp(forecast_start), periods=HORIZON, freq='h')
    try:
        aware = local.tz_localize(SOURCE_TZ, ambiguous='infer', nonexistent='shift_forward')
    except ValueError:
        aware = local.tz_localize(SOURCE_TZ, ambiguous=False, nonexistent='shift_forward')
    return aware.tz_convert('UTC').tz_localize(None)


def cycle_and_leads(forecast_start):
    target = local_horizon_to_utc(forecast_start)
    # Deliberately conservative as-of rule: previous calendar day 18Z.
    cycle = pd.Timestamp(forecast_start).normalize() - pd.Timedelta(hours=6)
    lo = int(math.floor(float((target[0] - cycle) / pd.Timedelta(hours=1)) / 3) * 3)
    hi = int(math.ceil(float((target[-1] - cycle) / pd.Timedelta(hours=1)) / 3) * 3)
    return cycle, list(range(lo, hi + 1, 3))


def _gfs_filename(cycle, lead_h):
    cycle = pd.Timestamp(cycle).tz_localize(None)
    return f'gfs.0p25.{cycle:%Y%m%d%H}.f{int(lead_h):03d}.grib2'


def build_ncss_url(cycle, lead_h):
    cycle = pd.Timestamp(cycle).tz_localize(None)
    date = cycle.strftime('%Y%m%d')
    fn = _gfs_filename(cycle, lead_h)
    return f'{NCSS_BASE}/{cycle:%Y}/{date}/{fn}?' + urlencode({
        'var': 'Temperature_height_above_ground',
        'vertCoord': 2,
        **BBOX,
        'horizStride': 1,
        'accept': 'netcdf',
    })


def build_osdf_url(cycle, lead_h):
    cycle = pd.Timestamp(cycle).tz_localize(None)
    date = cycle.strftime('%Y%m%d')
    return f'{OSDF_BASE}/{cycle:%Y}/{date}/{_gfs_filename(cycle, lead_h)}'


def _temperature_var(ds):
    if 'Temperature_height_above_ground' in ds.data_vars:
        return 'Temperature_height_above_ground'
    candidates = [v for v in ds.data_vars if 'Temperature' in v and 'height_above_ground' in v]
    if not candidates:
        raise RuntimeError(f'missing temp {list(ds.data_vars)}')
    return candidates[0]


def _to_fahrenheit(values, units):
    units_l = (units or '').lower().strip()
    x = np.asarray(values, dtype=float)
    if units_l == 'k' or 'kelvin' in units_l or (not units_l and np.nanmedian(x) > 150):
        return (x - 273.15) * 9 / 5 + 32
    if units_l == 'c' or 'celsius' in units_l or 'degc' in units_l:
        return x * 9 / 5 + 32
    if units_l == 'f' or 'fahrenheit' in units_l:
        return x
    if np.nanmedian(x) > 150:
        return (x - 273.15) * 9 / 5 + 32
    raise RuntimeError(units)


def _request_with_retries(session, url, attempts=8, timeout=90, sleep_fn=time.sleep):
    """Legacy NCSS request helper, retained as an audit/fallback path."""
    for attempt in range(attempts):
        try:
            r = session.get(url, timeout=timeout)
            if 400 <= r.status_code < 500:
                raise RuntimeError(f'NCSS {r.status_code}: {r.text[:500]} URL={url}')
            if r.status_code >= 500:
                if attempt == attempts - 1:
                    r.raise_for_status()
                sleep_fn(min(30, 2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except RuntimeError:
            raise
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt == attempts - 1:
                raise
            sleep_fn(min(30, 2 ** attempt))
    raise RuntimeError('unreachable retry state')


def _request_range_with_retries(
    session,
    url,
    start,
    end,
    attempts=6,
    timeout=60,
    sleep_fn=time.sleep,
):
    """Read a byte range from OSDF, retrying transient HTTP and stream failures."""
    transient = (
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )
    headers = {'Range': f'bytes={int(start)}-{int(end)}'}
    for attempt in range(attempts):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if 400 <= r.status_code < 500:
                raise RuntimeError(f'OSDF {r.status_code}: {r.text[:500]} URL={url}')
            if r.status_code >= 500:
                if attempt == attempts - 1:
                    r.raise_for_status()
                sleep_fn(min(20, 2 ** attempt))
                continue
            r.raise_for_status()
            if r.status_code != 206:
                raise RuntimeError(f'OSDF range not honored: {r.status_code} URL={url}')
            return r
        except RuntimeError:
            raise
        except transient:
            if attempt == attempts - 1:
                raise
            sleep_fn(min(20, 2 ** attempt))
    raise RuntimeError('unreachable range retry state')


def _grib2_product_info(header):
    """Return (total_len, discipline, category, number, surface_type, level)."""
    if len(header) < 16 or header[:4] != b'GRIB':
        return None
    total_len = int.from_bytes(header[8:16], 'big')
    if total_len < 16:
        return None
    discipline = header[6]
    pos = 16
    while pos + 5 <= len(header):
        section_len = int.from_bytes(header[pos:pos + 4], 'big')
        section_number = header[pos + 4]
        if section_len < 5:
            return None
        if section_number == 4:
            if pos + 28 > len(header):
                return None
            template = int.from_bytes(header[pos + 7:pos + 9], 'big')
            category = header[pos + 9]
            number = header[pos + 10]
            surface_type = header[pos + 22] if template in (0, 1, 2, 8, 11, 12, 15) else None
            scale = header[pos + 23] if surface_type is not None else None
            scaled_value = int.from_bytes(header[pos + 24:pos + 28], 'big') if surface_type is not None else None
            level = None
            if surface_type is not None and scale != 255 and scaled_value != 0xFFFFFFFF:
                signed_scale = scale if scale < 128 else scale - 256
                level = scaled_value * (10 ** (-signed_scale))
            return total_len, discipline, category, number, surface_type, level
        pos += section_len
    return None


def _is_2t_info(info):
    if info is None:
        return False
    _, discipline, category, number, surface_type, level = info
    return (
        discipline == 0
        and category == 0
        and number == 0
        and surface_type == 103
        and level is not None
        and abs(float(level) - 2.0) < 1e-9
    )


def _find_2t_header_in_bytes(chunk, chunk_start):
    """Find a 2m-temperature GRIB message whose header begins inside a range chunk."""
    pos = 0
    while True:
        idx = chunk.find(b'GRIB', pos)
        if idx < 0:
            return None
        info = _grib2_product_info(chunk[idx:idx + 4096])
        if _is_2t_info(info):
            return int(chunk_start) + idx, int(info[0])
        pos = idx + 4


def _file_size_from_content_range(response):
    value = response.headers.get('Content-Range', '')
    if '/' not in value:
        raise RuntimeError(f'missing Content-Range for OSDF response: {value!r}')
    return int(value.rsplit('/', 1)[-1])


def _scan_2t_header_by_message(session, cache_url, first_response=None, max_messages=1200):
    """Slow but deterministic fallback: walk GRIB message headers from byte zero."""
    offset = 0
    for message_index in range(1, max_messages + 1):
        if offset == 0 and first_response is not None:
            r = first_response
        else:
            r = _request_range_with_retries(session, cache_url, offset, offset + 4095)
        info = _grib2_product_info(r.content[:4096])
        if info is None:
            # Rare defensive expansion if Section 4 is beyond the first 4 KB.
            r = _request_range_with_retries(session, cache_url, offset, offset + 16383)
            info = _grib2_product_info(r.content[:16384])
        if info is None:
            raise RuntimeError(f'cannot parse GRIB header at offset={offset} index={message_index}')
        if _is_2t_info(info):
            return offset, int(info[0]), message_index
        offset += int(info[0])
    raise RuntimeError(f'2m temperature not found within {max_messages} GRIB messages')


def locate_2t_message(session, origin_url):
    """Locate 2t using a small range near the observed stable offset; fall back to header scan."""
    first = _request_range_with_retries(session, origin_url, 0, 4095, timeout=90)
    cache_url = first.url
    file_size = _file_size_from_content_range(first)
    center = int(round(file_size * OSDF_TARGET_RATIO))

    for window_bytes in OSDF_FAST_WINDOWS:
        half = window_bytes // 2
        start = max(0, center - half)
        end = min(file_size - 1, center + half - 1)
        r = _request_range_with_retries(session, cache_url, start, end, timeout=90)
        found = _find_2t_header_in_bytes(r.content, start)
        if found is not None:
            offset, message_len = found
            return {
                'cache_url': cache_url,
                'file_size': file_size,
                'offset': offset,
                'message_length': message_len,
                'mode': f'fast_{window_bytes // (1024 * 1024)}mb',
            }

    offset, message_len, message_index = _scan_2t_header_by_message(
        session,
        cache_url,
        first_response=first,
    )
    return {
        'cache_url': cache_url,
        'file_size': file_size,
        'offset': offset,
        'message_length': message_len,
        'mode': f'fallback_message_{message_index}',
    }


def _decode_station_values_from_grib(message):
    # Lazy import keeps pure contract tests light; operational Actions install eccodes.
    from eccodes import codes_get, codes_get_array, codes_new_from_message, codes_release

    gid = codes_new_from_message(message)
    try:
        short_name = str(codes_get(gid, 'shortName'))
        level_type = str(codes_get(gid, 'typeOfLevel'))
        level = float(codes_get(gid, 'level'))
        if short_name != '2t' or level_type != 'heightAboveGround' or abs(level - 2.0) > 1e-9:
            raise RuntimeError(f'unexpected GRIB field {short_name} {level_type} {level}')
        values = np.asarray(codes_get_array(gid, 'values'), dtype=float)
        lats = np.asarray(codes_get_array(gid, 'latitudes'), dtype=float)
        lons = np.asarray(codes_get_array(gid, 'longitudes'), dtype=float)
        units = str(codes_get(gid, 'units'))
        out = {}
        for name, (lat, lon) in STATIONS.items():
            qlon = lon % 360 if np.nanmin(lons) >= 0 else lon
            lon_dist = np.abs(lons - qlon)
            if np.nanmin(lons) >= 0:
                lon_dist = np.minimum(lon_dist, 360 - lon_dist)
            dist = (lats - lat) ** 2 + lon_dist ** 2
            i = int(np.nanargmin(dist))
            out[name] = float(_to_fahrenheit(np.array([values[i]]), units)[0])
        return out
    finally:
        codes_release(gid)


def fetch_station_values_osdf(cycle, lead_h, session, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{pd.Timestamp(cycle):%Y%m%d%H}_f{int(lead_h):03d}.2t.grib2'
    if cache_path.exists() and cache_path.stat().st_size >= 100:
        return _decode_station_values_from_grib(cache_path.read_bytes()), 'cache'

    origin_url = build_osdf_url(cycle, lead_h)
    located = locate_2t_message(session, origin_url)
    start = int(located['offset'])
    end = start + int(located['message_length']) - 1
    r = _request_range_with_retries(session, located['cache_url'], start, end, timeout=120)
    if len(r.content) != int(located['message_length']):
        raise RuntimeError(
            f'incomplete 2t GRIB message: got={len(r.content)} expected={located["message_length"]}'
        )
    cache_path.write_bytes(r.content)
    return _decode_station_values_from_grib(r.content), str(located['mode'])


# Legacy THREDDS/NCSS functions are intentionally retained for provenance comparison,
# but the operational run now uses OSDF byte-range extraction because 2017 THREDDS
# requests currently return persistent nginx 504s.
def fetch_grid(cycle, lead_h, session, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{cycle:%Y%m%d%H}_f{lead_h:03d}.nc'
    if not path.exists() or path.stat().st_size < 100:
        r = _request_with_retries(session, build_ncss_url(cycle, lead_h))
        if len(r.content) < 100:
            raise RuntimeError('tiny response')
        path.write_bytes(r.content)
    return xr.open_dataset(path, engine='netcdf4')


def station_values(ds):
    da = ds[_temperature_var(ds)].squeeze(drop=True)
    lat_name = next(n for n in da.coords if 'lat' in n.lower())
    lon_name = next(n for n in da.coords if 'lon' in n.lower())
    lons = np.asarray(da[lon_name].values, dtype=float)
    use360 = np.nanmin(lons) >= 0
    units = str(da.attrs.get('units', ''))
    out = {}
    for name, (lat, lon) in STATIONS.items():
        out[name] = float(_to_fahrenheit(
            np.asarray(da.sel({lat_name: lat, lon_name: (lon % 360 if use360 else lon)}, method='nearest').values),
            units,
        ).reshape(-1)[0])
    return out


def acquire_window_weather(forecast_start, cache_dir):
    cycle, leads = cycle_and_leads(forecast_start)
    session = requests.Session()
    rows = []
    for lead in leads:
        values, mode = fetch_station_values_osdf(cycle, lead, session, cache_dir)
        rows.append({'timestamp_utc': cycle + pd.Timedelta(hours=lead), **values})
        print('GFS_OSDF', cycle, 'lead', lead, 'mode', mode, 'ok', flush=True)
    series = pd.DataFrame(rows).set_index('timestamp_utc').sort_index()
    target = local_horizon_to_utc(forecast_start)
    series = series.reindex(series.index.union(target)).interpolate(method='time').reindex(target)
    out = series.reset_index()
    out.insert(0, 'timestamp_local', pd.date_range(pd.Timestamp(forecast_start), periods=HORIZON, freq='h'))
    return out


def zone_weather(station_df, variant):
    out = pd.DataFrame({'timestamp_local': pd.to_datetime(station_df['timestamp_local'])})
    for zone, weights in MAPPING_VARIANTS[variant].items():
        values = np.zeros(len(station_df))
        for station, weight in weights.items():
            values += float(weight) * pd.to_numeric(station_df[station]).to_numpy(float)
        out[zone] = values
    return out


def evaluate_windows(windows, outdir, seed=20260829):
    df = _load_fev_long()
    rows = []
    cache_dir = outdir / 'gfs_cache'
    first = next(iter(df.groupby('id', sort=True)))[1].sort_values('timestamp').reset_index(drop=True)
    cuts0 = exact_cut_indices(len(first))
    station_by_window = {}
    for window in windows:
        station_by_window[window] = acquire_window_weather(
            pd.Timestamp(first.iloc[cuts0[window - 1]]['timestamp']),
            cache_dir,
        )

    for sid, group in df.groupby('id', sort=True):
        zone = normalize_zone(str(sid))
        group = group.sort_values('timestamp').reset_index(drop=True)
        cuts = exact_cut_indices(len(group))
        supervised = _supervised(group)
        columns = safe_feature_columns(True)
        for window in windows:
            cut = cuts[window - 1]
            train = supervised.iloc[:cut]
            good = train[columns + ['target']].dropna().index
            X_train = train.loc[good, columns]
            y_train = train.loc[good, 'target'].astype(float)
            oracle = supervised.iloc[cut:cut + HORIZON].copy()
            truth = oracle['target'].astype(float).to_numpy()
            station = station_by_window[window]
            operational_tests = {}
            for variant in MAPPING_VARIANTS:
                weather = zone_weather(station, variant)[zone].to_numpy(float)
                test = oracle.copy()
                test.loc[:, 'weather'] = weather
                test.loc[:, 'weather_sq'] = weather ** 2
                test.loc[:, 'weather_hour'] = weather * test['hour_sin'].to_numpy(float)
                operational_tests[variant] = test[columns]

            for model_name in ('HGB', 'RF', 'ET'):
                model = model_factory(model_name, seed)
                model.fit(X_train, y_train)
                oracle_mae = float(np.mean(np.abs(truth - model.predict(oracle[columns]))))
                for variant in MAPPING_VARIANTS:
                    rows.append({
                        'zone': zone,
                        'window': window,
                        'mapping': variant,
                        'condition': 'ORACLE_WEATHER',
                        'model': model_name,
                        'mae': oracle_mae,
                    })
                    rows.append({
                        'zone': zone,
                        'window': window,
                        'mapping': variant,
                        'condition': 'OPERATIONAL_GFS',
                        'model': model_name,
                        'mae': float(np.mean(np.abs(
                            truth - model.predict(operational_tests[variant])
                        ))),
                    })
            print('DONE', zone, window, flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(outdir / 'operational_mae_by_window.csv', index=False)
    return result


def summarize(result, outdir):
    pivot = result.pivot_table(
        index=['zone', 'window', 'mapping', 'model'],
        columns='condition',
        values='mae',
    ).reset_index()
    pivot['D_operational_minus_oracle'] = pivot['OPERATIONAL_GFS'] - pivot['ORACLE_WEATHER']
    pivot.to_csv(outdir / 'operational_contrasts.csv', index=False)

    rank_rows = []
    for (mapping, zone, window), group in result.groupby(['mapping', 'zone', 'window']):
        matrix = group.pivot_table(index='condition', columns='model', values='mae')
        models = ['HGB', 'RF', 'ET']
        oracle_order = tuple(sorted(models, key=lambda x: (matrix.loc['ORACLE_WEATHER', x], x)))
        operational_order = tuple(sorted(models, key=lambda x: (matrix.loc['OPERATIONAL_GFS', x], x)))
        oracle_values = sorted(float(matrix.loc['ORACLE_WEATHER', x]) for x in models)
        rank_rows.append({
            'mapping': mapping,
            'zone': zone,
            'window': window,
            'rank_changed': oracle_order != operational_order,
            'winner_changed': oracle_order[0] != operational_order[0],
            'oracle_margin': oracle_values[1] - oracle_values[0],
        })
    pd.DataFrame(rank_rows).to_csv(outdir / 'operational_ranking.csv', index=False)


def parse_windows(value):
    out = []
    for part in value.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seed', type=int, default=20260829)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summarize(evaluate_windows(parse_windows(args.windows), outdir, args.seed), outdir)


if __name__ == '__main__':
    main()
