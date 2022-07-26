import pytest
from wave_equations import wave_equation


def test_WaveEquation_init_fails():
  "Wavelet class should be purely abstract so will fail if initiailized"
  with pytest.raises(TypeError):
    wave_equation.WaveEquation()
