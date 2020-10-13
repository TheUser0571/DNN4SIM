depends = ('ITKPyBase', 'ITKIOImageBase', 'ITKCommon', )
templates = (
  ('GDCMImageIOFactory', 'itk::GDCMImageIOFactory', 'itkGDCMImageIOFactory', True),
  ('GDCMSeriesFileNames', 'itk::GDCMSeriesFileNames', 'itkGDCMSeriesFileNames', True),
  ('GDCMImageIOEnums', 'itk::GDCMImageIOEnums', 'itkGDCMImageIOEnums', False),
  ('GDCMImageIO', 'itk::GDCMImageIO', 'itkGDCMImageIO', True),
)
snake_case_functions = ('gdcm_series_file_names', )
