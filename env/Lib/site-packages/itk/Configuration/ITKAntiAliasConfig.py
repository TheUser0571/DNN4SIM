depends = ('ITKPyBase', 'ITKLevelSets', )
templates = (
  ('AntiAliasBinaryImageFilter', 'itk::AntiAliasBinaryImageFilter', 'itkAntiAliasBinaryImageFilterIF2IF2', True, 'itk::Image< float,2 >, itk::Image< float,2 >'),
  ('AntiAliasBinaryImageFilter', 'itk::AntiAliasBinaryImageFilter', 'itkAntiAliasBinaryImageFilterIF3IF3', True, 'itk::Image< float,3 >, itk::Image< float,3 >'),
  ('AntiAliasBinaryImageFilter', 'itk::AntiAliasBinaryImageFilter', 'itkAntiAliasBinaryImageFilterID2ID2', True, 'itk::Image< double,2 >, itk::Image< double,2 >'),
  ('AntiAliasBinaryImageFilter', 'itk::AntiAliasBinaryImageFilter', 'itkAntiAliasBinaryImageFilterID3ID3', True, 'itk::Image< double,3 >, itk::Image< double,3 >'),
)
snake_case_functions = ('anti_alias_binary_image_filter', )
