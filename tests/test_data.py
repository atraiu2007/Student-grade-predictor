from src.data_generator import FEATURES, TARGET, generate_dataset


def test_generated_dataset_shape(tmp_path):
    path = tmp_path / "students.csv"
    df = generate_dataset(path, n=50)
    assert len(df) == 50
    assert all(col in df.columns for col in FEATURES + [TARGET])
    assert path.exists()
