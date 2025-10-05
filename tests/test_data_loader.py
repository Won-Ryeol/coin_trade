import pandas as pd
import numpy as np
import pytest

from coin_trade.data import load_random_data


def _write_csv(path, start, periods=16, freq='15T'):
    index = pd.date_range(start=start, periods=periods, freq=freq)
    base = np.linspace(100, 101, periods)
    df = pd.DataFrame(
        {
            'timestamp': index,
            'open': base,
            'high': base + 0.2,
            'low': base - 0.2,
            'close': base + 0.1,
            'volume': np.full(periods, 1000.0),
        }
    )
    df.to_csv(path, index=False)


def test_load_random_data_deterministic(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for i in range(3):
        _write_csv(data_dir / f'asset_{i}.csv', start=f'2024-01-0{i+1} 00:00:00')

    df1, files1 = load_random_data(str(data_dir), sample_k=2, random_seed=42, min_rows=10)
    df2, files2 = load_random_data(str(data_dir), sample_k=2, random_seed=42, min_rows=10)

    assert [f.name for f in files1] == [f.name for f in files2]
    assert not df1.empty
    assert df1.index.is_monotonic_increasing


def test_load_random_data_nested_directories(tmp_path):
    data_dir = tmp_path / 'data'
    (data_dir / 'KRW-BTC').mkdir(parents=True)
    (data_dir / 'KRW-ETH').mkdir(parents=True)
    _write_csv(data_dir / 'KRW-BTC' / '20240101.csv', start='2024-01-01 00:00:00')
    _write_csv(data_dir / 'KRW-ETH' / '20240102.csv', start='2024-01-02 00:00:00')

    df, files = load_random_data(str(data_dir), sample_k=2, random_seed=0, min_rows=10)
    assert len(files) == 2
    assert {f.parent.name for f in files} == {'KRW-BTC', 'KRW-ETH'}
    assert not df.empty


def test_load_random_data_invalid_sample(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_csv(data_dir / 'asset.csv', start='2024-01-01 00:00:00')

    with pytest.raises(ValueError):
        load_random_data(str(data_dir), sample_k=2)


def test_load_random_data_invalid_frequency(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    index = pd.date_range('2024-01-01 00:00:00', periods=10, freq='30T')
    df = pd.DataFrame(
        {
            'timestamp': index,
            'open': 1.0,
            'high': 1.1,
            'low': 0.9,
            'close': 1.05,
            'volume': 1000,
        }
    )
    df.to_csv(data_dir / 'bad.csv', index=False)

    with pytest.raises(ValueError):
        load_random_data(str(data_dir), min_rows=10)

def test_load_random_data_skips_large_gap(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_csv(data_dir / 'asset_0.csv', start='2024-01-01 00:00:00')
    _write_csv(data_dir / 'asset_1.csv', start='2024-01-02 00:00:00')
    _write_csv(data_dir / 'asset_far.csv', start='2024-03-10 00:00:00')

    df, files = load_random_data(str(data_dir), sample_k=2, random_seed=0, min_rows=10)

    assert [f.name for f in files] == ['asset_0.csv', 'asset_1.csv']
    max_gap = df.index.to_series().diff().dropna().max()
    assert max_gap <= pd.Timedelta(days=1)
