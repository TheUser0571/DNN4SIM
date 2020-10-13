depends = ('ITKPyBase', 'ITKImageFilterBase', 'ITKFiniteDifference', )
templates = (
  ('CurvatureFlowImageFilter', 'itk::CurvatureFlowImageFilter', 'itkCurvatureFlowImageFilterIF2IF2', True, 'itk::Image< float,2 >, itk::Image< float,2 >'),
  ('CurvatureFlowImageFilter', 'itk::CurvatureFlowImageFilter', 'itkCurvatureFlowImageFilterIF3IF3', True, 'itk::Image< float,3 >, itk::Image< float,3 >'),
  ('CurvatureFlowImageFilter', 'itk::CurvatureFlowImageFilter', 'itkCurvatureFlowImageFilterID2ID2', True, 'itk::Image< double,2 >, itk::Image< double,2 >'),
  ('CurvatureFlowImageFilter', 'itk::CurvatureFlowImageFilter', 'itkCurvatureFlowImageFilterID3ID3', True, 'itk::Image< double,3 >, itk::Image< double,3 >'),
  ('MinMaxCurvatureFlowImageFilter', 'itk::MinMaxCurvatureFlowImageFilter', 'itkMinMaxCurvatureFlowImageFilterIF2IF2', True, 'itk::Image< float,2 >, itk::Image< float,2 >'),
  ('MinMaxCurvatureFlowImageFilter', 'itk::MinMaxCurvatureFlowImageFilter', 'itkMinMaxCurvatureFlowImageFilterIF3IF3', True, 'itk::Image< float,3 >, itk::Image< float,3 >'),
  ('MinMaxCurvatureFlowImageFilter', 'itk::MinMaxCurvatureFlowImageFilter', 'itkMinMaxCurvatureFlowImageFilterID2ID2', True, 'itk::Image< double,2 >, itk::Image< double,2 >'),
  ('MinMaxCurvatureFlowImageFilter', 'itk::MinMaxCurvatureFlowImageFilter', 'itkMinMaxCurvatureFlowImageFilterID3ID3', True, 'itk::Image< double,3 >, itk::Image< double,3 >'),
  ('BinaryMinMaxCurvatureFlowImageFilter', 'itk::BinaryMinMaxCurvatureFlowImageFilter', 'itkBinaryMinMaxCurvatureFlowImageFilterIF2IF2', True, 'itk::Image< float,2 >, itk::Image< float,2 >'),
  ('BinaryMinMaxCurvatureFlowImageFilter', 'itk::BinaryMinMaxCurvatureFlowImageFilter', 'itkBinaryMinMaxCurvatureFlowImageFilterIF3IF3', True, 'itk::Image< float,3 >, itk::Image< float,3 >'),
  ('BinaryMinMaxCurvatureFlowImageFilter', 'itk::BinaryMinMaxCurvatureFlowImageFilter', 'itkBinaryMinMaxCurvatureFlowImageFilterID2ID2', True, 'itk::Image< double,2 >, itk::Image< double,2 >'),
  ('BinaryMinMaxCurvatureFlowImageFilter', 'itk::BinaryMinMaxCurvatureFlowImageFilter', 'itkBinaryMinMaxCurvatureFlowImageFilterID3ID3', True, 'itk::Image< double,3 >, itk::Image< double,3 >'),
)
snake_case_functions = ('binary_min_max_curvature_flow_image_filter', 'curvature_flow_image_filter', 'min_max_curvature_flow_image_filter', )
