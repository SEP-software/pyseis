import inspect
import numpy as np
import copy
from pyseis.wave_equations import acoustic_isotropic, elastic_isotropic
from pyseis.wavelets.acoustic import Acoustic2D, Acoustic3D
from pyseis.wavelets.elastic import Elastic2D, Elastic3D
from pyseis.geometries import Geometry

WAVE_EQUATION_MODULES = [acoustic_isotropic, elastic_isotropic]
WAVE_EQUATION_TYPES = {}
for WAVE_EQUATION_MODULE in WAVE_EQUATION_MODULES:
  WAVE_EQUATION_TYPES.update({
      name: value
      for name, value in inspect.getmembers(WAVE_EQUATION_MODULE)
      if inspect.isclass(value)
  })

WAVELET_MODULES = [Acoustic2D, Acoustic3D, Elastic2D, Elastic3D]
WAVELET_TYPES = {}
for WAVELET_MODULE in WAVELET_MODULES:
  WAVELET_TYPES.update({
      name: value
      for name, value in inspect.getmembers(WAVELET_MODULE)
      if inspect.isclass(value)
  })

GEOMETRY_TYPES = {
    name: value
    for name, value in inspect.getmembers(Geometry)
    if inspect.isclass(value)
}


class Survey():
  REQUIRED_CONFIGS = ['geometry', 'wavelet', 'wave_equation']

  def __init__(self, model: np.ndarray, config: dict):
    for required_config in self.REQUIRED_CONFIGS:
      if required_config not in config:
        raise RuntimeError(f'config missing {required_config}.')

    self.config = copy.deepcopy(config)
    self.geometry = self._make_geometry(self.config['geometry'])
    self.wavelet = self._make_wavelet(self.config['wavelet'])

    self.config['wave_equation']['args']['model'] = model
    self.config['wave_equation']['args']['wavelet'] = self.wavelet.get_arr()
    self.config['wave_equation']['args']['d_t'] = self.wavelet.d_t
    self.config['wave_equation']['args'][
        'src_locations'] = self.geometry.get_src_positions()
    self.config['wave_equation']['args'][
        'rec_locations'] = self.geometry.get_rec_positions()
    self.wave_equation = self._make_wave_equation(self.config['wave_equation'])

  def _make_geometry(self, config_geometry):
    geometry_type = config_geometry['type']
    geometry_args = config_geometry['args']
    if geometry_type not in GEOMETRY_TYPES:
      raise RuntimeError(f'{geometry_type} is not a supported acqusition type')
    return GEOMETRY_TYPES[geometry_type](**geometry_args)

  def _make_wave_equation(self, config_wave_equation):
    wave_equation_type = config_wave_equation['type']
    wave_equation_args = config_wave_equation['args']
    if wave_equation_type not in WAVE_EQUATION_TYPES:
      raise RuntimeError(
          f'{wave_equation_type} is not a supported wave equation type')
    wave_equation = WAVE_EQUATION_TYPES[wave_equation_type](
        **wave_equation_args)

    return wave_equation

  def _make_wavelet(self, config_wavelet):
    wavelet_type = config_wavelet['type']
    wavelet_args = config_wavelet['args']
    if wavelet_type not in WAVELET_TYPES:
      raise RuntimeError(f'{wavelet_type} is not a supported wavelet type')
    return WAVELET_TYPES[wavelet_type](**wavelet_args)

  def fwd(self, model):
    data = self.wave_equation.fwd(model)
    return data


def parse_yaml(yaml_path):
  """Given a path to a .yaml file, parses into a dictionary.

  Args:
      config_fn (str or Path): .yaml to be parsed

  Returns:
      dictionary: parsed yaml.
  """
  with open(yaml_path, 'r') as stream:
    parsed_yaml = yaml.safe_load(stream)
  return parsed_yaml


def main(yaml_path):
  # parse yaml
  yaml_path = Path(yaml_path)
  parsed_yaml = pipeline.parse_yaml(yaml_path)

  # read in model
  input_model_fn = parsed_yaml['input_model']

  # get output data filename
  output_data_fn = parsed_yaml['output_data']

  # read in configs
  config = {}
  config['geometry'] = parsed_yaml['geometry']
  config['wavelet'] = parsed_yaml['wavelet']
  config['wave_equation'] = parsed_yaml['wave_equation']

  # initialize survey
  test_survey = survey.Survey(model, config)

  #save data


if __name__ == "__main__":
  # parse command line arguements
  parser = argparse.ArgumentParser()
  parser.add_argument('yaml_path',
                      type=str,
                      help='(string): Path of yaml instructions file.')
  args = parser.parse_args()

  # run main
  main(args.yaml_path)
